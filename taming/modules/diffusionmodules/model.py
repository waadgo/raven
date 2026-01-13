# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch, warnings, functools, os

def term_color(text, c):   # quick & dirty ANSI colours
    codes = dict(red=31, green=32, yellow=33, cyan=36)
    return f"\033[{codes[c]}m{text}\033[0m"

class FlashCompatMixin:
    """
    Adds `_flash_ok` property and `_explain_flash()` helper.
    Call `self._explain_flash()` once per module (e.g. in `__init__`).
    """
    def _flash_capable(self, head_dim, dtype):
        cc_ok   = torch.cuda.get_device_capability()[0] >= 8
        dtype_ok= dtype in (torch.float16, torch.bfloat16)
        dim_ok  = (head_dim <= 128) and (head_dim % 8 == 0)
        return cc_ok and dtype_ok and dim_ok

    def _explain_flash(self, head_dim, num_heads, dtype):
        if not torch.cuda.is_available():
            print(term_color("→ FlashAttention not available (CPU run).", "yellow"))
            return False

        flash = self._flash_capable(head_dim, dtype)
        kernel = torch.backends.cuda.preferred_linalg_library
        if flash:
            print(term_color(
                f"✓ Flash-SDP kernel will be used  "
                f"(h={num_heads}, d={head_dim}, {dtype}, {kernel})", "green"))
        else:
            why = []
            cc = torch.cuda.get_device_capability()[0]
            if cc < 8:          why.append(f"SM{cc*10} GPU")
            if dtype not in (torch.float16, torch.bfloat16): why.append(f"dtype={dtype}")
            if head_dim > 128:  why.append(f"head_dim={head_dim}>128")
            if head_dim % 8:    why.append(f"head_dim%8={head_dim%8}")
            msg = " / ".join(why)
            print(term_color(f"→ Flash disabled, falling back to efficient/math ({msg})",
                             "red"))
        return flash

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0,1,0,0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv):
        """
        3D version of your Upsample block.
        Assumes input shape [batch_size, in_channels, D, H, W].
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )

    def forward(self, x):
        # For 3D data, use 3D interpolation
        # scale_factor=(2,2,2) will upsample D, H, W all by factor of 2
        x = F.interpolate(x, scale_factor=(2, 2, 2), mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels, with_conv):
        """
        3D version of your Downsample block.
        Assumes input shape [batch_size, in_channels, D, H, W].
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # No asymmetric padding in torch Conv3d, must do it ourselves
            # kernel_size=3, stride=2, padding=0
            self.conv = nn.Conv3d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=2,
                padding=0
            )

    def forward(self, x):
        if self.with_conv:
            # 3D padding tuple: (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
            # e.g. we add 1 unit of padding to right, bottom, and back if needed
            pad = (0, 1, 0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # If no conv, just average-pool with kernel_size=2
            x = F.avg_pool3d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
    
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w,z = q.shape
        q = q.reshape(b,c,h*w*z)
        q = q.permute(0,2,1)   # b,hwz,c
        k = k.reshape(b,c,h*w*z) # b,c,hwz
        w_ = torch.bmm(q,k)     # b,hwz,hwz    w[b,i,j,k]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w*z)
        w_ = w_.permute(0,2,1)   # b,hwz,hwz (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hwz (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w,z)

        h_ = self.proj_out(h_)

        return x+h_
    
class Model(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, t=None):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Model3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, use_timestep=True):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock3D(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, t=None):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

def rescale_to_uniform_voxel_size(x0_0, orig_zoom, iz_h=1.0, iz_w=1.0):
    def ceil_next_even(f):
        if f == 0:
            return 32  # Minimum dimension
        else:
            return 32 * torch.ceil(f / 32.)

    batch_size, filtnum, h, w = x0_0.size()

    # Lists to store results
    onemm_slices = []
    inner_network_shape_perbs = []
    padding_info = []  # Store padding values

    # Loop through each batch member
    for bs in range(batch_size):
        # Get the original zoom factors for each dimension
        orig_vox_h, orig_vox_w, _ = orig_zoom[bs, :]

        # Compute rescaling factors inversely proportional to iz_h and iz_w
        factor_h = orig_vox_h / iz_h  # Inverse relationship
        factor_w = orig_vox_w / iz_w  # Inverse relationship

        # Compute the target size for the current batch member using ceil_next_even
        H_norm = int(ceil_next_even(h * factor_h).item())  # Convert tensor to scalar
        W_norm = int(ceil_next_even(w * factor_w).item())  # Convert tensor to scalar

        # Resize current slice to the new normalized size
        resized_slice = F.interpolate(x0_0[bs:bs+1], size=(H_norm, W_norm), mode='bicubic', align_corners=True)

        # Clamp values between 0 and 1
        resized_slice = torch.clamp(resized_slice, 0, 1)
        onemm_slices.append(resized_slice)
        inner_network_shape_perbs.append((H_norm, W_norm))

    # Compute the maximum normalized height and width to zero-pad slices
    H_norm_max = max([shape[0] for shape in inner_network_shape_perbs])
    W_norm_max = max([shape[1] for shape in inner_network_shape_perbs])

    # Create a tensor to store the padded slices
    x0_00 = torch.zeros((batch_size, filtnum, H_norm_max, W_norm_max), device=x0_0.device)

    # Zero-pad each resized slice to match the max dimensions and store padding info
    for bs in range(batch_size):
        resized_slice = onemm_slices[bs]
        _, _, H_norm, W_norm = resized_slice.size()

        # Compute padding values
        pad_h = (H_norm_max - H_norm) // 2
        pad_w = (W_norm_max - W_norm) // 2

        # Pad the slice
        x0_00[bs] = F.pad(resized_slice, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)

        # Store padding information for this batch element
        padding_info.append((pad_h, pad_h, pad_w, pad_w))

    # Return the padded tensor, inner shape per batch, and the padding info
    return x0_00, inner_network_shape_perbs, padding_info

def invert_rescale_to_uniform_voxel_size(x0_00, inner_network_shape_perbs, padding_info, output_shape):
    batch_size, filtnum, H_norm_max, W_norm_max = x0_00.size()
    
    # List to store the unpadded and resized slices
    unpadded_slices = []

    # Loop through each batch member to remove padding and resize
    for bs in range(batch_size):
        # Get the padding information and original inner shape for this batch member
        pad_h, _, pad_w, _ = padding_info[bs]
        H_norm, W_norm = inner_network_shape_perbs[bs]

        # Calculate the start and end indices for slicing, based on padding
        start_h = pad_h
        end_h = start_h + H_norm

        start_w = pad_w
        end_w = start_w + W_norm

        # Slice the tensor based on the computed indices
        unpadded_slice = x0_00[bs, :, start_h:end_h, start_w:end_w]

        # Resize back to the original dimensions using output_shape
        unpadded_resized_slice = F.interpolate(unpadded_slice.unsqueeze(0), size=output_shape, mode='bicubic', align_corners=True)

        # Store the final result
        unpadded_slices.append(unpadded_resized_slice)

    # Concatenate all unpadded and resized slices into a final tensor
    return torch.cat(unpadded_slices, dim=0)

class EncoderVINN(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 inner_res = 0.7,
                 in_channels, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.inner_res = inner_res

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        self.conv_vinn = torch.nn.Conv2d(self.ch,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = inner_res
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: # WATCH OUT! ATTN RESOLUTIONS ARE NOW IN TERMS OF VOXEL SIZE OF FEATURE MAPS
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, orig_zoom):
        #orig_zoom is a list of the zooms [zh, zw, zz]
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None
        shapes = x.shape
        h = shapes[2]
        w = shapes[3]
        output_shape = (h, w)
        # downsampling
        hin = self.conv_in(x) #hin are tensors of shape [1, 128, 256, 256]
        hin, inner_network_shape_perbs, padding_info = rescale_to_uniform_voxel_size(hin, orig_zoom, iz_h = self.inner_res, iz_w = self.inner_res)
        hs = [self.conv_vinn(hin)]
        for i_level in range(self.num_resolutions): #num of resolutions is 5
            for i_block in range(self.num_res_blocks):# i_block iterates from 0 to num of resolutions (5)
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, inner_network_shape_perbs, padding_info, output_shape
    
class DecoderVINN(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, 
                 inner_res = 0.7,
                 in_channels, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.inner_res = inner_res
        
        

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = inner_res * 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape = {} mm.".format(self.z_shape))
        print(f'The internal resolution of the network (VINN) is: {inner_res} mm isotropic ..')
        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        
        self.conv_out_vinn = torch.nn.Conv2d(block_in,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, inner_network_shape_perbs, padding_info, output_shape):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        #Bringing feature maps to native space:
        h = invert_rescale_to_uniform_voxel_size(h, inner_network_shape_perbs, padding_info, output_shape)
        h = self.conv_out_vinn(h)
        h = self.conv_out(h)
        return h

    
class Encoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)] #hs is a list of tensors of shape [1, 128, 256, 256]
        for i_level in range(self.num_resolutions): #num of resolutions is 5
            for i_block in range(self.num_res_blocks):# i_block iterates from 0 to num of resolutions (5)
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VUNet(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 in_channels, c_channels,
                 resolution, z_channels, use_timestep=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList([
                torch.nn.Linear(self.ch,
                                self.temb_ch),
                torch.nn.Linear(self.temb_ch,
                                self.temb_ch),
            ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(c_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.z_in = torch.nn.Conv2d(z_channels,
                                    block_in,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=2*block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x, z):
        #assert x.shape[2] == x.shape[3] == self.resolution

        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        z = self.z_in(z)
        h = torch.cat((h,z),dim=1)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__()
        self.model = nn.ModuleList([nn.Conv2d(in_channels, in_channels, 1),
                                     ResnetBlock(in_channels=in_channels,
                                                 out_channels=2 * in_channels,
                                                 temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=2 * in_channels,
                                                out_channels=4 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     ResnetBlock(in_channels=4 * in_channels,
                                                out_channels=2 * in_channels,
                                                temb_channels=0, dropout=0.0),
                                     nn.Conv2d(2*in_channels, in_channels, 1),
                                     Upsample(in_channels, with_conv=True)])
        # end
        self.norm_out = Normalize(in_channels)
        self.conv_out = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i in [1,2,3]:
                x = layer(x, None)
            else:
                x = layer(x)

        h = self.norm_out(x)
        h = nonlinearity(h)
        x = self.conv_out(h)
        return x


class UpsampleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, ch, num_res_blocks, resolution,
                 ch_mult=(2,2), dropout=0.0):
        super().__init__()
        # upsampling
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = in_channels
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            res_block = []
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                res_block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
            self.res_blocks.append(nn.ModuleList(res_block))
            if i_level != self.num_resolutions - 1:
                self.upsample_blocks.append(Upsample(block_in, True))
                curr_res = curr_res * 2

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # upsampling
        h = x
        for k, i_level in enumerate(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.res_blocks[i_level][i_block](h, None)
            if i_level != self.num_resolutions - 1:
                h = self.upsample_blocks[k](h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Encoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)


    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)] #3D Input of shape expected [BS, C, D, H, W]
        for i_level in range(self.num_resolutions): #num of resolutions is 5
            for i_block in range(self.num_res_blocks):# i_block iterates from 0 to num of resolutions (5)
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) #Shape torch.Size([1, 2, 48, 48, 8]) GPU: 11133MiB / 23028MiB
        return h


class Decoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Assuming Normalize and nonlinearity are defined earlier in the file
# from .model import Normalize, nonlinearity # Or similar import

class ResnetBlock3DFiLM(nn.Module):
    """
    3D ResNet block modified to incorporate FiLM conditioning. (Corrected)
    Uses nn.SiLU() in Sequential, nonlinearity() in forward.
    """
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, cond_embed_dim=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.cond_embed_dim = cond_embed_dim

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None and temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None # Explicitly set to None if not used

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # FiLM generation layer
        if self.cond_embed_dim is not None and self.cond_embed_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.SiLU(), # Correct module for Sequential
                nn.Linear(self.cond_embed_dim, 2 * self.out_channels)
            )
            if len(self.cond_proj) > 1 and isinstance(self.cond_proj[-1], nn.Linear):
                 nn.init.zeros_(self.cond_proj[-1].weight)
                 nn.init.zeros_(self.cond_proj[-1].bias)
        else:
            self.cond_proj = None # Ensure it's None if not used

        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None
            self.nin_shortcut = None


    def forward(self, x, temb, cond_emb=None):
        h = x
        h = self.conv1( nonlinearity( self.norm1(h) ) )

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h_norm = self.norm2(h)

        if cond_emb is not None and self.cond_proj is not None:
            gamma_beta = self.cond_proj(cond_emb)[:, :, None, None, None]
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
            h_modulated = h_norm * (1. + gamma) + beta
        else:
            h_modulated = h_norm

        h = self.conv2( self.dropout( nonlinearity(h_modulated) ) )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x_shortcut = self.conv_shortcut(x)
            elif self.nin_shortcut is not None:
                x_shortcut = self.nin_shortcut(x)
            else: # Should not happen if in_channels != out_channels, but safety
                x_shortcut = x
        else:
            x_shortcut = x

        return x_shortcut + h

class Encoder3D_aniso(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4), num_res_blocks=2,
                 attn_resolutions=(), dropout=0.0, resamp_with_conv=True, in_channels=1,
                 resolution=192, depth=32, z_channels=1, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.depth = depth
        self.in_channels = in_channels

        # initial conv
        self.conv_in = nn.Conv3d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res_h = resolution
        curr_res_w = resolution
        curr_res_d = depth
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in,
                                           out_channels=block_out,
                                           temb_channels=self.temb_ch,
                                           dropout=dropout))
                block_in = block_out
                if curr_res_h in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample3D_HW(block_in, resamp_with_conv)
                curr_res_h //= 2
                curr_res_w //= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in, out_channels=block_in,
                                         temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in, out_channels=block_in,
                                         temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, 2 * z_channels if double_z else z_channels,
                                  kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder3D_aniso(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1, 2, 4), num_res_blocks=2,
                 attn_resolutions=(), dropout=0.0, resamp_with_conv=True, in_channels=1,
                 resolution=192, depth=32, z_channels=1, give_pre_end=False, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.depth = depth
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_h = resolution // 2 ** (self.num_resolutions - 1)
        curr_w = resolution // 2 ** (self.num_resolutions - 1)
        curr_d = depth  # unchanged

        self.z_shape = (1, z_channels, curr_d, curr_h, curr_w)

        # z to block_in
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in, out_channels=block_in,
                                         temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in, out_channels=block_in,
                                         temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock3D(in_channels=block_in, out_channels=block_out,
                                           temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if (resolution // 2 ** i_level) in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample3D_HW(block_in, resamp_with_conv)
                curr_h *= 2
                curr_w *= 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        if self.give_pre_end:
            return h
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Encoder3DFiLM(nn.Module):
    """
    3D Encoder using ResnetBlock3DFiLM for voxel size conditioning.
    Conditions on SOURCE voxel size.
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True,
                 cond_input_dim=3, cond_embed_dim=512, # FiLM parameters for SOURCE cond
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0 # No time embedding in VAE encoder typically
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.cond_input_dim = cond_input_dim
        self.cond_embed_dim = cond_embed_dim

        print(f"Encoder3DFiLM: Conditioning on input dim {self.cond_input_dim}, embedding to {self.cond_embed_dim}")

        # Conditioning Embedder
        if self.cond_embed_dim is not None and self.cond_embed_dim > 0:
            self.cond_embedder = nn.Sequential(
                nn.Linear(self.cond_input_dim, self.ch * 4),
                nn.SiLU(),
                nn.Linear(self.ch * 4, self.cond_embed_dim)
            )
        else:
            print("Encoder3DFiLM: No conditioning embedding will be used.")
            self.cond_embed_dim = None
            self.cond_embedder = None

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3DFiLM(in_channels=block_in, # Use FiLM block
                                                out_channels=block_out,
                                                temb_channels=self.temb_ch,
                                                dropout=dropout,
                                                cond_embed_dim=self.cond_embed_dim)) # Pass cond dim
                block_in = block_out
                 # Calculate resolution at this level correctly BEFORE checking attn_resolutions
                level_res = resolution // 2**(i_level)
                if level_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in)) # Keep standard self-attention

            down = nn.Module()
            down.block = block
            down.attn = attn # Store attention modules
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                # Update curr_res based on spatial dimensions if needed
                curr_res = level_res // 2 # Assuming isotropic downsampling
            self.down.append(down)

        # Middle block
        # block_in should be the output channels from the last downsampling level
        block_in = ch*ch_mult[-1] # Correctly get channels for mid block
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3DFiLM(in_channels=block_in, # Use FiLM block
                                             out_channels=block_in,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,
                                             cond_embed_dim=self.cond_embed_dim)
        self.mid.attn_1 = AttnBlock3D(block_in) # Standard self-attention
        self.mid.block_2 = ResnetBlock3DFiLM(in_channels=block_in, # Use FiLM block
                                             out_channels=block_in,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,
                                             cond_embed_dim=self.cond_embed_dim)

        # End block
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        2*z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, cond_input=None):
        # Generate conditioning embedding
        cond_emb = None
        if self.cond_embedder is not None and cond_input is not None:
            # Ensure cond_input is a tensor
            if not isinstance(cond_input, torch.Tensor):
                # Create tensor explicitly on the SAME DEVICE as input x
                cond_input = torch.tensor(cond_input, device=x.device, dtype=torch.float32)
            # Ensure correct dtype
            if cond_input.dtype != torch.float32:
                cond_input = cond_input.to(dtype=torch.float32)
            # <<< ADDED/MODIFIED: Ensure correct device >>>
            if cond_input.device != x.device:
                cond_input = cond_input.to(x.device)

            # Handle batch dimension
            if cond_input.ndim == 1:
                 cond_input = cond_input.unsqueeze(0).expand(x.shape[0], -1)
            elif cond_input.shape[0] != x.shape[0]:
                 raise ValueError(f"Batch size mismatch: x {x.shape[0]}, cond {cond_input.shape[0]}")

            cond_emb = self.cond_embedder(cond_input) # Now 

        temb = None # No time embedding

        # Downsampling
        hs = [self.conv_in(x)] # Store feature maps at each resolution
        h = hs[-1] # Current feature map

        for i_level in range(self.num_resolutions):
            # Apply ResNet blocks for this level
            for i_block in range(self.num_res_blocks):
                res_block = self.down[i_level].block[i_block]
                h = res_block(h, temb, cond_emb=cond_emb) # Update h in place

            # Apply Attention blocks after ResNet blocks for this level
            if len(self.down[i_level].attn) > 0:
                for attn_block in self.down[i_level].attn:
                    h = attn_block(h) # Update h in place

            hs.append(h) # Store the output of this level (before downsampling)

            # Downsample if not the last level
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h) # Downsample for next level's input
                # We actually store the output *before* downsampling in hs for skip connections
                # The input to the next level is the downsampled 'h'

        # Middle block
        # 'h' now holds the output of the last downsampling operation
        h = self.mid.block_1(h, temb, cond_emb=cond_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, cond_emb=cond_emb)

        # End block
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class Decoder3DFiLM(nn.Module):
    """
    3D Decoder using ResnetBlock3DFiLM for voxel size conditioning. (Corrected)
    """
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False,
                 cond_input_dim=3, cond_embed_dim=512, # FiLM parameters
                 **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0 # No time embedding in VAE decoder typically
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.cond_input_dim = cond_input_dim
        self.cond_embed_dim = cond_embed_dim

        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        # Adjust z_shape if resolution is not cubic
        # Assuming resolution is isotropic for simplicity here
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)
        print(f"Decoder3DFiLM: z_shape approx {self.z_shape}, dims {np.prod(self.z_shape)}")
        print(f"Decoder3DFiLM: Conditioning on input dim {self.cond_input_dim}, embedding to {self.cond_embed_dim}")

        # Conditioning Embedder
        if self.cond_embed_dim is not None and self.cond_embed_dim > 0:
            self.cond_embedder = nn.Sequential(
                nn.Linear(self.cond_input_dim, self.ch * 4),
                nn.SiLU(), # <<< CORRECT MODULE for Sequential
                nn.Linear(self.ch * 4, self.cond_embed_dim)
            )
        else:
            print("Decoder3DFiLM: No conditioning embedding will be used.")
            self.cond_embed_dim = None
            self.cond_embedder = None

        # Input convolution
        self.conv_in = nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle blocks
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3DFiLM(in_channels=block_in, out_channels=block_in,
                                             temb_channels=self.temb_ch, dropout=dropout,
                                             cond_embed_dim=self.cond_embed_dim)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3DFiLM(in_channels=block_in, out_channels=block_in,
                                             temb_channels=self.temb_ch, dropout=dropout,
                                             cond_embed_dim=self.cond_embed_dim)

        # Upsampling blocks
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            # Correct calculation of block_in for upsampling path
            # It should be the output channels from the previous level (or mid block)
            current_block_in = block_in # block_in holds channels from previous level

            for i_block in range(self.num_res_blocks+1):
                # ResNet block input channels = channels from previous layer in this level
                block.append(ResnetBlock3DFiLM(in_channels=current_block_in, # Use correct input channels
                                                 out_channels=block_out,
                                                 temb_channels=self.temb_ch,
                                                 dropout=dropout,
                                                 cond_embed_dim=self.cond_embed_dim))
                current_block_in = block_out # Update block_in for the next block in *this* level

            # block_in now holds the final output channels for this level (block_out)
            block_in = block_out # Update block_in for the *next* (lower index) level's input

            up = nn.Module()
            up.block = block
            up.attn = nn.ModuleList() # Initialize attn list
            # Calculate resolution at this level correctly BEFORE checking attn_resolutions
            level_res = resolution // 2**(i_level)
            if level_res in attn_resolutions:
                 up.attn.append(AttnBlock3D(block_in)) # Add attention if resolution matches

            if i_level != 0:
                # Upsample takes block_in (output channels of this level) as input channels
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                # Update curr_res correctly (assuming isotropic for now)
                curr_res = level_res * 2

            self.up.insert(0, up) # Prepend

        # Output layers
        self.norm_out = Normalize(block_in) # norm_out uses the final block_in channels
        self.conv_out = nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)


    def forward(self, z, cond_input=None):
        # ... (other setup) ...
        # Generate conditioning embedding
        temb = None
        cond_emb = None
        if self.cond_embedder is not None and cond_input is not None:
            # Ensure cond_input is a tensor
            if not isinstance(cond_input, torch.Tensor):
                # Create tensor explicitly on the SAME DEVICE as input z
                cond_input = torch.tensor(cond_input, device=z.device, dtype=torch.float32)
            # Ensure correct dtype
            if cond_input.dtype != torch.float32:
                cond_input = cond_input.to(dtype=torch.float32)
            # <<< ADDED/MODIFIED: Ensure correct device >>>
            if cond_input.device != z.device:
                cond_input = cond_input.to(z.device)

            # Handle batch dimension
            if cond_input.ndim == 1:
                cond_input = cond_input.unsqueeze(0).expand(z.shape[0], -1)
            elif cond_input.shape[0] != z.shape[0]:
                 raise ValueError(f"Batch size mismatch: z {z.shape[0]}, cond {cond_input.shape[0]}")

            cond_emb = self.cond_embedder(cond_input) # Now 

        # Initial convolution
        h = self.conv_in(z)

        # Middle blocks
        h = self.mid.block_1(h, temb, cond_emb=cond_emb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb, cond_emb=cond_emb)

        # Upsampling pathway
        for i_level in reversed(range(self.num_resolutions)):
            # Apply ResNet blocks for this level
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb, cond_emb=cond_emb)

            # Apply Attention blocks for this level
            if len(self.up[i_level].attn) > 0:
                 for attn_block in self.up[i_level].attn: # Iterate through attn blocks if multiple
                      h = attn_block(h)

            # Upsample if not the last level
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # Final output processing
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h) # <<< Using nonlinearity FUNCTION here is FINE
        h = self.conv_out(h)
        return h

class Encoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, resizing_pos=None, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.resizing_pos = resizing_pos
        if self.resizing_pos is None:
            print("`resizing_pos` not found in ddconfig. Defaulting to downsampling at all but the last level.")
            self.resizing_pos = [1] * (self.num_resolutions - 1) + [0]

        # ───────────────────────────────────────────────────────────────────
        #   NEW: Validation to ensure list lengths match
        # ───────────────────────────────────────────────────────────────────
        if len(self.resizing_pos) != self.num_resolutions:
            raise ValueError(
                f"Configuration Error: The length of 'resizing_pos' ({len(self.resizing_pos)}) "
                f"must be equal to the length of 'ch_mult' ({self.num_resolutions})."
            )
        # ───────────────────────────────────────────────────────────────────

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if self.resizing_pos[i_level] == 1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in, 2*z_channels if double_z else z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        temb = None
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if hasattr(self.down[i_level], 'downsample'):
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder3D(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, resizing_pos=None, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        self.resizing_pos = resizing_pos
        if self.resizing_pos is None:
            print("`resizing_pos` not found in ddconfig. Defaulting to upsampling at all but the first level.")
            self.resizing_pos = [1] * (self.num_resolutions - 1) + [0]

        # ───────────────────────────────────────────────────────────────────
        #   NEW: Validation to ensure list lengths match
        # ───────────────────────────────────────────────────────────────────
        if len(self.resizing_pos) != self.num_resolutions:
            raise ValueError(
                f"Configuration Error: The length of 'resizing_pos' ({len(self.resizing_pos)}) "
                f"must be equal to the length of 'ch_mult' ({self.num_resolutions})."
            )
        # ───────────────────────────────────────────────────────────────────

        in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        num_downsamples = sum(self.resizing_pos)
        curr_res = resolution // (2**num_downsamples)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        self.conv_in = torch.nn.Conv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0 and self.resizing_pos[i_level-1] == 1:
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape
        temb = None
        h = self.conv_in(z)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if hasattr(self.up[i_level], 'upsample'):
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

# from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

# ---------------------------------------------------------------------
# Small helpers carried over from your code base
# ---------------------------------------------------------------------
def nonlinearity(x):           # SiLU / Swish
    return x * torch.sigmoid(x)

class Normalize(nn.GroupNorm): # identical to your original
    def __init__(self, num_channels):
        super().__init__(num_groups=32, num_channels=num_channels, eps=1e-6, affine=True)

# ---------------------------------------------------------------------
#  MEMORY-FRIENDLY FLASH-ATTENTION IN 3-D
# ---------------------------------------------------------------------
from torch.nn.functional import scaled_dot_product_attention as sdpa
# from utils_flash import FlashCompatMixin

class Attention3D_v2(nn.Module, FlashCompatMixin):
    """
    [B,C,D,H,W] → Flash/efficient/math-SDP → residual, with
    automatic num_heads adjustment to satisfy head_dim ≤ 128 ∧ 8 | head_dim.
    """
    def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        # --- pick num_heads so head_dim is a multiple of 8 and ≤128 ----------
        while (in_channels // num_heads) > 128 or (in_channels // num_heads) % 8:
            num_heads *= 2                                  # fall back to more heads
        self.num_heads, self.head_dim = num_heads, in_channels // num_heads
        if in_channels % num_heads:
            raise ValueError(f"{in_channels=} not divisible by {num_heads=}")

        inner = in_channels
        self.norm      = Normalize(in_channels)
        self.qkv       = nn.Conv3d(in_channels, inner * 3, 1)
        self.proj_out  = nn.Conv3d(inner,      in_channels, 1)
        self.dropout_p = dropout

        # Inform user once
        if not hasattr(Attention3D_v2, "_banner"):
            dtype = torch.float16  # typical; real dtype known only after to(device)
            self._explain_flash(self.head_dim, self.num_heads, dtype)
            Attention3D_v2._banner = True

    # ------------------------------------------------------------------
    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(self.norm(x)).reshape(
            B, 3, self.num_heads, self.head_dim, -1)        # -1 = S=D·H·W
        q, k, v = (t.permute(0,1,3,2) for t in qkv)         # B,h,S,d

        attn = sdpa(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.,
            is_causal=False
        )                                                   # B,h,S,d
        attn = attn.permute(0,1,3,2).contiguous().view(B, C, D, H, W)
        return x + self.proj_out(attn)                                           # residual

# ---------------------------------------------------------------------
#  BUILDING BLOCKS (unchanged logic, only suffixed)
# ---------------------------------------------------------------------
class ResnetBlock3D_v2(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1, self.norm2 = Normalize(in_channels), Normalize(out_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        # skip connection if in!=out
        if in_channels != out_channels:
            self.conv_shortcut = (nn.Conv3d(in_channels, out_channels, 3, padding=1)
                                  if conv_shortcut
                                  else nn.Conv3d(in_channels, out_channels, 1))

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None:
            h += self.temb_proj(nonlinearity(temb)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)
        return x + h

class Upsample3D_v2(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, 3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=(2,2,2), mode="nearest")
        return self.conv(x) if self.with_conv else x

class Downsample3D_v2(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv3d(in_channels, in_channels, 3, stride=2)
    def forward(self, x):
        if self.with_conv:
            # pad so D,H,W divisible by 2
            x = F.pad(x, (0,1,0,1,0,1))
            x = self.conv(x)
        else:
            x = F.avg_pool3d(x, 2, stride=2)
        return x

class Normalize(nn.Module):
    """
    A normalization layer, typically GroupNorm.
    It robustly adjusts num_groups to be a divisor of num_channels.
    """
    def __init__(self, num_channels, num_groups=32):
        super().__init__()
        # Input validation for num_channels and num_groups
        if not isinstance(num_channels, int) or num_channels < 0:
            raise ValueError("num_channels must be a non-negative integer.")
        if not isinstance(num_groups, int) or num_groups <= 0:
            raise ValueError("num_groups must be a positive integer.")

        if num_channels == 0: # Edge case: no normalization needed or possible for 0 channels
            self.norm = nn.Identity()
        elif num_channels < num_groups or num_channels % num_groups != 0:
            # If num_channels is small (e.g., less than num_groups),
            # or not divisible by num_groups, find a suitable num_groups.
            if num_channels <= num_groups:
                # Use num_channels as num_groups if it's smaller or equal.
                # This means each channel is its own group.
                num_groups = num_channels
            else:
                # Find the largest valid number of groups <= preferred num_groups
                # that divides num_channels.
                possible_num_groups = [g for g in range(1, min(num_groups, num_channels) + 1) if num_channels % g == 0]
                if not possible_num_groups: # Should not happen if g=1 is always possible
                    num_groups = 1 # Fallback: normalize over all channels as a single group
                else:
                    num_groups = max(possible_num_groups)
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)
        else:
            # num_channels is divisible by num_groups
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-6, affine=True)

    def forward(self, x):
        # Skip normalization if input channels is 0 (e.g. empty tensor placeholder)
        if x.shape[1] == 0:
            return x
        return self.norm(x)

class CrossResolutionAttention3D(nn.Module):
    """
    Attention mechanism where the original 3D input (queries) attends to
    a downsampled version of itself (keys and values), using
    torch.nn.functional.scaled_dot_product_attention.
    """
    def __init__(self, in_channels, pool_type='max', downsample_factor=2, num_norm_groups=32, attention_dropout_p=0.0):
        super().__init__()

        # Input validation
        if not isinstance(in_channels, int) or in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if not isinstance(downsample_factor, int) or downsample_factor < 1:
            raise ValueError("downsample_factor must be an integer >= 1.")
        if pool_type not in ['max', 'avg']:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Choose 'max' or 'avg'.")
        if not (0.0 <= attention_dropout_p < 1.0):
            raise ValueError("attention_dropout_p must be between 0.0 (inclusive) and 1.0 (exclusive).")


        self.in_channels = in_channels
        self.downsample_factor = downsample_factor
        self.attention_dropout_p = attention_dropout_p

        self.norm = Normalize(in_channels, num_groups=num_norm_groups)

        # Projections for Q from original resolution feature map
        self.q_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # Pooling layer
        if self.downsample_factor > 1:
            if pool_type == 'max':
                self.pool = nn.MaxPool3d(kernel_size=downsample_factor, stride=downsample_factor)
            elif pool_type == 'avg': # pool_type == 'avg'
                self.pool = nn.AvgPool3d(kernel_size=downsample_factor, stride=downsample_factor)
        else: # downsample_factor is 1, no actual pooling needed
            self.pool = nn.Identity()

        # Projections for K and V from the downsampled feature map
        self.k_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        # Output projection
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # b: batch_size, c: channels, h,w,z: spatial dimensions (e.g., Depth, Height, Width)
        b, c, h, w, z = x.shape

        if c != self.in_channels:
            raise ValueError(f"Input channels {c} does not match model's initialized in_channels {self.in_channels}")

        # 1. Normalize original input
        x_norm = self.norm(x)

        # 2. Generate Queries (Q) from original resolution
        # q_spatial shape: (b, c, h, w, z)
        q_spatial = self.q_conv(x_norm)
        num_orig_voxels = h * w * z
        # Reshape Q for attention: (b, N_orig, c) where N_orig = h*w*z
        # (b, c, N_orig) -> (b, N_orig, c)
        query = q_spatial.view(b, c, num_orig_voxels).permute(0, 2, 1)

        # 3. Create downsampled version for Keys (K) and Values (V)
        x_downsampled_norm = self.pool(x_norm) # Applies pooling if downsample_factor > 1

        # k_spatial_downsampled shape: (b, c, h_d, w_d, z_d)
        k_spatial_downsampled = self.k_conv(x_downsampled_norm)
        # v_spatial_downsampled shape: (b, c, h_d, w_d, z_d)
        v_spatial_downsampled = self.v_conv(x_downsampled_norm)

        _, _, h_d, w_d, z_d = k_spatial_downsampled.shape # Dimensions of downsampled map
        num_down_voxels = h_d * w_d * z_d

        # Reshape K for attention: (b, N_down, c)
        # (b, c, N_down) -> (b, N_down, c)
        key = k_spatial_downsampled.view(b, c, num_down_voxels).permute(0, 2, 1)

        # Reshape V for attention: (b, N_down, c)
        # (b, c, N_down) -> (b, N_down, c)
        value = v_spatial_downsampled.view(b, c, num_down_voxels).permute(0, 2, 1)

        # 4. Compute attention using scaled_dot_product_attention
        # query shape: (b, num_orig_voxels, c)
        # key shape:   (b, num_down_voxels, c)
        # value shape: (b, num_down_voxels, c)
        # Output shape: (b, num_orig_voxels, c)
        attn_output_reshaped = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,  # No mask typically needed for this non-causal cross-attention
            dropout_p=self.attention_dropout_p if self.training else 0.0, # Apply dropout only during training
            is_causal=False
        )

        # 5. Reshape attended output back to original spatial dimensions
        # (b, N_orig, c) -> (b, c, N_orig) -> (b, c, h, w, z)
        attn_output_spatial = attn_output_reshaped.permute(0, 2, 1).contiguous().view(b, c, h, w, z)

        # 6. Final projection
        projected_output = self.proj_out(attn_output_spatial)

        # 7. Residual connection
        return x + projected_output
# ---------------------------------------------------------------------
#  ENCODER / DECODER
# ---------------------------------------------------------------------
class Encoder3D_v3(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0., resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, resizing_pos=None, **ignore):
        super().__init__()
        self.ch, self.temb_ch = ch, 0
        n_levels = len(ch_mult)
        self.num_res_blocks, self.resolution, self.in_channels = num_res_blocks, resolution, in_channels

        self.resizing_pos = resizing_pos or ([1]*(n_levels-1) + [0])
        assert len(self.resizing_pos) == n_levels, "`resizing_pos` length mismatch."

        self.conv_in = nn.Conv3d(in_channels, ch, 3, padding=1)
        curr_res, in_ch_mult = resolution, (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()

        for lvl in range(n_levels):
            block, attn = nn.ModuleList(), nn.ModuleList()
            in_ch  = ch * in_ch_mult[lvl]
            out_ch = ch * ch_mult[lvl]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock3D_v2(in_channels=in_ch, out_channels=out_ch,
                                              temb_channels=self.temb_ch, dropout=dropout))
                in_ch = out_ch
                if curr_res in attn_resolutions:
                    attn.append(CrossResolutionAttention3D(in_ch))
            down_l = nn.Module(); down_l.block, down_l.attn = block, attn
            if self.resizing_pos[lvl]:
                down_l.downsample = Downsample3D_v2(in_ch, resamp_with_conv)
                curr_res //= 2
            self.down.append(down_l)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D_v2(in_channels=in_ch, dropout=dropout)
        self.mid.attn_1 = CrossResolutionAttention3D(in_ch)
        self.mid.block_2 = ResnetBlock3D_v2(in_channels=in_ch, dropout=dropout)

        # output
        self.norm_out = Normalize(in_ch)
        self.conv_out = nn.Conv3d(in_ch, 2*z_channels if double_z else z_channels, 3, padding=1)

    def forward(self, x):
        temb, hs = None, [self.conv_in(x)]
        for lvl in range(len(self.down)):
            for blk in self.down[lvl].block:
                h = blk(hs[-1], temb); hs.append(h)
                if self.down[lvl].attn: h = self.down[lvl].attn[len(hs)-2](h); hs[-1] = h
            if hasattr(self.down[lvl], "downsample"):
                hs.append(self.down[lvl].downsample(hs[-1]))
        h = hs[-1]; h = self.mid.block_1(h, temb); h = self.mid.attn_1(h); h = self.mid.block_2(h, temb)
        return self.conv_out(nonlinearity(self.norm_out(h)))

class Decoder3D_v3(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0., resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, resizing_pos=None, **kw):
        super().__init__()
        self.ch, self.temb_ch = ch, 0
        n_levels = len(ch_mult)
        self.num_res_blocks, self.give_pre_end = num_res_blocks, give_pre_end
        self.resizing_pos = resizing_pos or ([1]*(n_levels-1)+[0])
        assert len(self.resizing_pos) == n_levels

        block_in = ch * ch_mult[-1]
        num_down = sum(self.resizing_pos)
        curr_res = resolution // (2 ** num_down)
        self.z_shape = (1, z_channels, curr_res, curr_res, curr_res)

        self.conv_in = nn.Conv3d(z_channels, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D_v2(in_channels=block_in, dropout=dropout)
        self.mid.attn_1 = CrossResolutionAttention3D(block_in)
        self.mid.block_2 = ResnetBlock3D_v2(in_channels=block_in, dropout=dropout)

        self.up = nn.ModuleList()
        for lvl in reversed(range(n_levels)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[lvl]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock3D_v2(in_channels=block_in, out_channels=block_out,
                                              temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CrossResolutionAttention3D(block_in))
            up_l = nn.Module(); up_l.block, up_l.attn = block, attn
            if lvl != 0 and self.resizing_pos[lvl-1]:
                up_l.upsample = Upsample3D_v2(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up_l)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv3d(block_in, out_ch, 3, padding=1)

    def forward(self, z):
        temb, h = None, self.conv_in(z)
        h = self.mid.block_1(h, temb); h = self.mid.attn_1(h); h = self.mid.block_2(h, temb)
        for lvl in reversed(range(len(self.up))):
            for blk in self.up[lvl].block:
                h = blk(h, temb)
                if self.up[lvl].attn: h = self.up[lvl].attn[0](h)
            if hasattr(self.up[lvl], "upsample"):
                h = self.up[lvl].upsample(h)
        if self.give_pre_end:
            return h
        return self.conv_out(nonlinearity(self.norm_out(h)))

