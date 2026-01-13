import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main_hdf5 import instantiate_from_config
import numpy as np
from taming.modules.diffusionmodules.model import Encoder, Decoder, EncoderVINN, DecoderVINN
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
import torch.nn.functional as F
import math
import random

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 trainconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.downscaling_factors = trainconfig['downscaling_factors']
        self.ae_mode = trainconfig['ae_mode'] 
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x): # [BATCH_SIZE, CHANNELS, SIZE, SIZE]
        h = self.encoder(x) # [BATCH_SIZE, 256, 16, 16]
        h = self.quant_conv(h) 
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def get_input(self, batch, k):
        
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()

    def downscale(self, x, shape, interpolation_method = 'trilinear', align_corners = True):
        """
        x is a torch tensor, will be converted to torch tensor, downscaled by upscaling factor, then converted back to numpy in lr and returned

        """
        
        if interpolation_method == 'area':
            lr = torch.nn.functional.interpolate(x.unsqueeze(1), size = shape,
                                                mode = interpolation_method)
        else:
            lr = torch.nn.functional.interpolate(x.unsqueeze(1), size = shape,
                                                mode = interpolation_method,
                                                align_corners = align_corners)
        return lr.squeeze(1)

    def upscale(self, x, shape, interpolation_method = 'trilinear', align_corners = True):
        """
        x of shape (BS, C, hin, win) will be reshaped to (BS, C, hout, wout) given shape=(hout, wout) using the given interpolation_method
        """
        if interpolation_method == 'area':
            hr = torch.nn.functional.interpolate(x.unsqueeze(1), size = shape, 
                                                mode = interpolation_method)
        else:
            hr = torch.nn.functional.interpolate(x.unsqueeze(1), size = shape, 
                                                mode = interpolation_method, 
                                                align_corners = align_corners)
        return hr.squeeze(1)

    def gaussian_kernel1d(self, sigma, size, device):
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x**2) / (2 * sigma**2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        # Determine the device of the input image
        device = img.device

        # Define kernel sizes
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)
        
        # Generate Gaussian kernels and move them to the correct device
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, size_z, 1, 1)
        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, 1, size_h, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, 1, size_w)
        
        # Convolve with Gaussian kernel along each dimension
        img = F.conv3d(img.unsqueeze(1), kernel_z, padding=(size_z // 2, 0, 0))
        img = F.conv3d(img, kernel_h, padding=(0, size_h // 2, 0))
        img = F.conv3d(img, kernel_w, padding=(0, 0, size_w // 2))
        
        return img.squeeze(1)

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma = 0.5):
        # Calculate sigmas for Gaussian blur
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0
        
        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)
        
        sigma_z = sigma_z * (1 + aug_z)
        sigma_h = sigma_h * (1 + aug_h)
        sigma_w = sigma_w * (1 + aug_w)
        
        # Apply Gaussian blur
        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train = True):
        """
        This function gets an object "img" of shape (BS, C, H, W), and three downscaling/upscaling factors named uf_h, uf_w, and uf_z.
        It returns an object "degraded_slice" of shape (BS, 1, H, W), and target_slice (BS, 1, H, W) that has gone through the following degradations:
            1) Blur the input img using random levels of gaussian blur (could be removed in the future)
            2) using a Hanning window with sinc interpolant downscaling img to shape (BS, C/uf_z, H/uf_h, W/uf_w)
            2) upscaling this low resolution tensor to shape (BS, C, H, W) using trilinear interpolation
            3) Extract the middle slice from the HR input and the degraded HR input. There should be a perfect correspondance to each other. 
        This is expected to mimic the behaviour of mincresample using the default configuration. 
        """
        # img = img.to(torch.bfloat16)
        img = img.to(torch.float)

        # #reshape from (bs, h, c, w) to (bs, c, h, w):
        # img = img.permute(0, 2, 1, 3)
        bs, c, h, w = img.shape 
        discard_slices = int((c % uf_z)) #Will discard the last "discard_slices" from the volume
        img = img[:, 0:c-discard_slices, :, :]
        bs, c, h, w = img.shape 
        slice_of_interest = c//2 #We are only interested in working with the middle slice. 
        target_slice = img[:, slice_of_interest, :, :].unsqueeze(1)
        # interpolation_methods = ['trilinear', 'sinc']
        interpolation_methods = ['trilinear']
        interpolation_method = np.random.choice(interpolation_methods)
        degradations = ['blur', 'noblur']
        degrade = np.random.choice(degradations)
        if degrade == 'blur':
            img = self.blur_tensor(img, uf_z = uf_z, uf_h = uf_h, uf_w = uf_w)
        # if interpolation_method == 'sinc':
        #     downscaled_vol = mincresample_torch_separable(img, uf_h, uf_w, uf_z)
        shape = (c//uf_z, h//uf_h, w//uf_w)
        downscaled_vol = self.downscale(img, shape, interpolation_method = interpolation_method, align_corners = True) #in-plane downscaling
        upscaled_vol = self.upscale(downscaled_vol, (c, h, w), interpolation_method = 'trilinear')
        degraded_slice = upscaled_vol[:, slice_of_interest, :, :].unsqueeze(1)
        return degraded_slice, target_slice

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        def weighted_choice(options):
            # Calculate probabilities
            n = len(options)
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)
        
        if is_train:
            # noise = random.choice(self.downscaling_factors)
            # noise = 'R' + str(noise)
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else:
            # noise = random.choice(self.downscaling_factors)
            # noise = 'R' + str(noise)
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        
        return uf_h, uf_w, uf_z

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key) # [BATCH_SIZE, CHANNELS, H, W]
        if self.ae_mode == 'gt2gt':
            xrec, qloss = self(x)
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, qloss = self(lres_ims)
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(qloss, lres_ims, xrec, optimizer_idx, self.global_step, #lres_ims instead of x, since we are trying to reconstruct lres_ims
                                                last_layer=self.get_last_layer(), split="train")

                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, lres_ims, xrec, optimizer_idx, self.global_step, #lres_ims instead of x, since we are trying to reconstruct lres_ims
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, qloss = self(lres_ims)
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(qloss, hres_ims, xrec, optimizer_idx, self.global_step, #hres_ims instead of x, since we are trying to reconstruct hres_ims
                                                last_layer=self.get_last_layer(), split="train")

                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            if optimizer_idx == 1:
                # discriminator
                discloss, log_dict_disc = self.loss(qloss, hres_ims, xrec, optimizer_idx, self.global_step, #hres_ims instead of x, since we are trying to reconstruct hres_ims
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss
        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        if self.ae_mode == 'gt2gt':
            xrec, qloss = self(x)
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, qloss = self(lres_ims)
            aeloss, log_dict_ae = self.loss(qloss, lres_ims, xrec, 0, self.global_step, #lres_ims instead of x, since we are trying to reconstruct lres_ims
                                                last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(qloss, lres_ims, xrec, 1, self.global_step, #lres_ims instead of x, since we are trying to reconstruct lres_ims
                                                last_layer=self.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, qloss = self(lres_ims)
            aeloss, log_dict_ae = self.loss(qloss, hres_ims, xrec, 0, self.global_step, #hres_ims instead of x, since we are trying to reconstruct hres_ims
                                                last_layer=self.get_last_layer(), split="val")

            discloss, log_dict_disc = self.loss(qloss, hres_ims, xrec, 1, self.global_step, #hres_ims instead of x, since we are trying to reconstruct hres_ims
                                                last_layer=self.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]
        else:
            raise NotImplementedError
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if self.ae_mode == 'gt2gt':
            xrec, _ = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, _ = self(lres_ims)
            if lres_ims.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(lres_ims)
                xrec = self.to_rgb(xrec)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(x.shape) == 3:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, _ = self(lres_ims)
            if lres_ims.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(lres_ims)
                xrec = self.to_rgb(xrec)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelVINN(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 zoom_key="zoom",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__()
        self.image_key = image_key
        self.zoom_key = zoom_key
        self.encoder = EncoderVINN(**ddconfig)
        self.decoder = DecoderVINN(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x, zoom): # [BATCH_SIZE, CHANNELS, SIZE, SIZE]
        h, inner_network_shape_perbs, padding_info, output_shape = self.encoder(x, zoom) # [BATCH_SIZE, 256, 16, 16]
        h = self.quant_conv(h) 
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info, inner_network_shape_perbs, padding_info, output_shape

    def decode(self, quant, inner_network_shape_perbs, padding_info, output_shape):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant, inner_network_shape_perbs, padding_info, output_shape)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, zoom):
        quant, diff, _, inner_network_shape_perbs, padding_info, output_shape = self.encode(input, zoom)
        dec = self.decode(quant, inner_network_shape_perbs, padding_info, output_shape)
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
    
    def get_orig_zooms(self, batch, k):
        output_voxsize = batch[k]
        return output_voxsize

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key) # [BATCH_SIZE, CHANNELS, H, W]
        zoom = self.get_orig_zooms(batch, self.zoom_key)
        xrec, qloss = self(x, zoom)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        zoom = self.get_orig_zooms(batch, self.zoom_key)
        xrec, qloss = self(x, zoom)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        zoom = self.get_orig_zooms(batch, self.zoom_key)
        x = x.to(self.device)
        xrec, _ = self(x, zoom)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x

class VQSegmentationModel(VQModel):
    def __init__(self, n_labels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("colorize", torch.randn(3, n_labels, 1, 1))

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return opt_ae

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        zoom = self.get_orig_zooms(batch, self.zoom_key)
        xrec, qloss = self(x, zoom)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="train")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return aeloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        zoom = self.get_orig_zooms(batch, self.zoom_key)
        xrec, qloss = self(x, zoom)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, split="val")
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        total_loss = log_dict_ae["val/total_loss"]
        self.log("val/total_loss", total_loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        return aeloss

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            # convert logits to indices
            xrec = torch.argmax(xrec, dim=1, keepdim=True)
            xrec = F.one_hot(xrec, num_classes=x.shape[1])
            xrec = xrec.squeeze(1).permute(0, 3, 1, 2).float()
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log


class VQNoDiscModel(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None
                 ):
        super().__init__(ddconfig=ddconfig, lossconfig=lossconfig, n_embed=n_embed, embed_dim=embed_dim,
                         ckpt_path=ckpt_path, ignore_keys=ignore_keys, image_key=image_key,
                         colorize_nlabels=colorize_nlabels)

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        # autoencode
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="train")
        output = pl.TrainResult(minimize=aeloss)
        output.log("train/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        return output

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, self.global_step, split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        output = pl.EvalResult(checkpoint_on=rec_loss)
        output.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output.log_dict(log_dict_ae)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


class GumbelVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 temperature_scheduler_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 kl_weight=1e-8,
                 remap=None,
                 ):

        z_channels = ddconfig["z_channels"]
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )

        self.loss.n_classes = n_embed
        self.vocab_size = n_embed

        self.quantize = GumbelQuantize(z_channels, embed_dim,
                                       n_embed=n_embed,
                                       kl_weight=kl_weight, temp_init=1.0,
                                       remap=remap)

        self.temperature_scheduler = instantiate_from_config(temperature_scheduler_config)   # annealing of temp

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def temperature_scheduling(self):
        self.quantize.temperature = self.temperature_scheduler(self.global_step)

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode_code(self, code_b):
        raise NotImplementedError

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.temperature_scheduling()
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log("temperature", self.quantize.temperature, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # encode
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, _, _ = self.quantize(h)
        # decode
        x_rec = self.decode(quant)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log


class EMAVQ(VQModel):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         n_embed,
                         embed_dim,
                         ckpt_path=None,
                         ignore_keys=ignore_keys,
                         image_key=image_key,
                         colorize_nlabels=colorize_nlabels,
                         monitor=monitor,
                         )
        self.quantize = EMAVectorQuantizer(n_embed=n_embed,
                                           embedding_dim=embed_dim,
                                           beta=0.25,
                                           remap=remap)
    def configure_optimizers(self):
        lr = self.learning_rate
        #Remove self.quantize from parameter list since it is updated via EMA
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []                                           