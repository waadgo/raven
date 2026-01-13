import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from main_hdf5 import instantiate_from_config
import numpy as np
from taming.modules.diffusionmodules.model import Encoder, Decoder, EncoderVINN, DecoderVINN, Encoder3D, Decoder3D, Encoder3D_v3, Decoder3D_v3
import torch.nn as nn
from taming.modules.diffusionmodules.model import Encoder3DFiLM, Decoder3DFiLM
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import VectorQuantizer3 as VectorQuantizer3D
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer
from taming.data.utils import volshow
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution, DiagonalGaussianDistribution3D
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
        self.current_mini_epoch = 0
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

    def on_save_checkpoint(self, checkpoint):
        # Save current_mini_epoch so we can resume exactly from the correct mini-epoch
        checkpoint['current_mini_epoch'] = self.current_mini_epoch

    def on_load_checkpoint(self, checkpoint):
        # Restore current_mini_epoch when resuming training
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)

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
            bs, c, h, w = x.shape 
            slice_of_interest = c//2 #We are only interested in working with the middle slice. 
            x = x[:, slice_of_interest, :, :].unsqueeze(1)
            
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
            bs, c, h, w = x.shape 
            slice_of_interest = c//2 #We are only interested in working with the middle slice. 
            x = x[:, slice_of_interest, :, :].unsqueeze(1)
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
            bs, c, h, w = x.shape 
            slice_of_interest = c//2 #We are only interested in working with the middle slice. 
            x = x[:, slice_of_interest, :, :].unsqueeze(1)
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

class VQModel3D(pl.LightningModule):
    def __init__(self,
                 ddconfig,            # config for 3D encoder/decoder
                 lossconfig,          # config for vector-quantizer + discriminator losses
                 trainconfig,         # e.g. downscaling_factors, ae_mode
                 n_embed,             # number of codebook embeddings
                 embed_dim,           # dimensionality of each embedding
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        """
        A 3D Vector-Quantized model using a 3D encoder & decoder. Works with 5D
        volumes of shape [BS, C, H, W, Z], optionally downscaling/upsampling
        or applying blur for degrade modes.

        Args:
          ddconfig: Dict with parameters for 3D encoder/decoder (z_channels, etc.)
          lossconfig: Dict to instantiate VQ loss, including vector quantizer config, disc config, etc.
          trainconfig: Dict with keys like 'downscaling_factors', 'ae_mode'
          n_embed: Size of the codebook (number of embeddings)
          embed_dim: Dimensionality of each codebook embedding
          ckpt_path: Optional checkpoint path
          ignore_keys: Keys to ignore when loading state_dict
          image_key: Key to extract from batch
          colorize_nlabels: For segmentation colorization (optional)
          monitor: Name of metric to monitor
        """
        super().__init__()
        self.current_mini_epoch = 0
        self.image_key = image_key
        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer3D(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, kernel_size=1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], kernel_size=1)
        self.downscaling_factors = trainconfig['downscaling_factors']
        self.ae_mode = trainconfig['ae_mode'] 

        if colorize_nlabels is not None:
            assert isinstance(colorize_nlabels, int)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        # Load checkpoint if provided
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def on_save_checkpoint(self, checkpoint):
        # Save current_mini_epoch so we can resume exactly from the correct mini-epoch
        checkpoint['current_mini_epoch'] = self.current_mini_epoch

    def on_load_checkpoint(self, checkpoint):
        # Restore current_mini_epoch when resuming training
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # ------------------ VQ ENCODE/DECODE ---------------------
    def encode(self, x):
        """
        Encode the 3D volume into latents, then pass through quant_conv for dimension matching.
        """
        h = self.encoder(x)                 # [BS, z_channels, H', W', Z']
        h = self.quant_conv(h)             # [BS, embed_dim,  H', W', Z']
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        """
        Decode from quantized latents -> post_quant_conv -> 3D decoder
        """
        quant = self.post_quant_conv(quant) # [BS, z_channels, H', W', Z']
        dec = self.decoder(quant)          # [BS, C, H, W, Z]
        return dec
    
    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, x):
        """
        Full VQ forward: encode -> vector-quantize -> decode
        Returns:
          dec: The reconstructed volume [BS, C, H, W, Z]
          qloss: The codebook commitment or VQ objective from the vector quantizer
          quant: The quantized latents
        """
        # 1) Encode to latents
        quant, diff, _ = self.encode(x)   # shape [BS, embed_dim, H', W', Z']
        dec = self.decode(quant)
        return dec, diff

    # ------------------ DATA UTILS (3D) ---------------------
    def get_input(self, batch, k):
        """
        Extracts a 5D volume [BS, C, H, W, Z] from the batch dict using key k.
        If it's 4D, we add a channel dimension.
        """
        x = batch[k]
        if len(x.shape) == 4:  # [BS, H, W, Z]
            x = x.unsqueeze(1)
        x = x.to(memory_format=torch.contiguous_format)
        return x

    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        """
        3D downscaling of x to 'shape' with F.interpolate.
        x shape: [BS, C, H, W, Z]
        shape: (H_out, W_out, Z_out)
        """
        if interpolation_method == 'area':
            lr = F.interpolate(x, size=shape, mode=interpolation_method)
        else:
            lr = F.interpolate(x, size=shape, mode=interpolation_method, align_corners=align_corners)
        return lr

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        """
        3D upscaling of x to 'shape' using F.interpolate.
        x shape: [BS, C, H, W, Z]
        shape: (H_out, W_out, Z_out)
        """
        if interpolation_method == 'area':
            hr = F.interpolate(x, size=shape, mode=interpolation_method)
        else:
            hr = F.interpolate(x, size=shape, mode=interpolation_method, align_corners=align_corners)
        return hr

    def gaussian_kernel1d(self, sigma, size, device):
        """
        Creates a 1D Gaussian kernel given a standard deviation (sigma) and kernel size.
        """
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x**2) / (2 * sigma**2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        """
        Applies a 3D Gaussian blur to a 5D tensor [BS, C, H, W, Z].
        Convolving along each dimension (H, W, Z) with 1D kernels.
        """
        device = img.device
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)

        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, size_h, 1, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, size_w, 1)
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, 1, 1, size_z)

        # 1) Convolve along H
        img = F.conv3d(img.unsqueeze(1), kernel_h, padding=(size_h // 2, 0, 0))
        # 2) Convolve along W
        img = F.conv3d(img, kernel_w, padding=(0, size_w // 2, 0))
        # 3) Convolve along Z
        img = F.conv3d(img, kernel_z, padding=(0, 0, size_z // 2))

        return img.squeeze(1)

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
        """
        Blurs a 5D tensor [BS, C, H, W, Z] using random per-dimension sigmas.
        """
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0

        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)

        sigma_z *= (1 + aug_z)
        sigma_h *= (1 + aug_h)
        sigma_w *= (1 + aug_w)

        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
        """
        Degrade a 5D volume [BS, C, H, W, Z] by optional blur + downscaling + upscaling.
        Returns (degraded_vol, target_vol).
        """
        target_vol = img
        bs, c, h, w, z = img.shape

        interpolation_methods = ['trilinear']
        interpolation_method = np.random.choice(interpolation_methods)
        degrade = np.random.choice(['blur', 'noblur'])

        if degrade == 'blur':
            img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        shape = (h // uf_h, w // uf_w, z // uf_z)
        downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
        upscaled_vol = self.upscale(downscaled_vol, (h, w, z), interpolation_method='trilinear')
        return upscaled_vol, target_vol

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        """
        Randomly pick downscaling factors from self.downscaling_factors, optionally coupling them.
        """
        def weighted_choice(options):
            n = len(options)
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)

        if is_train:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        return uf_h, uf_w, uf_z

    # ------------------ TRAINING & VALIDATION ------------------

    def training_step(self, batch, batch_idx, optimizer_idx):
        """
        Handles the degrade modes: gt2gt, deg2deg, deg2gt
        Then does forward pass:
           - If optimizer_idx=0 => generator (VQ model) update
           - If optimizer_idx=1 => discriminator update
        """
        inputs = self.get_input(batch, self.image_key)  # shape [BS, C, H, W, Z]
        # inputs = inputs.to(torch.half)

        # "gt2gt": no degrade, just feed inputs
        if self.ae_mode == 'gt2gt':
            reconstructions, qloss = self(inputs)
            if optimizer_idx == 0:
                # generator / autoencode update
                aeloss, log_dict_ae = self.loss(qloss, inputs, reconstructions,
                                                optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            elif optimizer_idx == 1:
                # discriminator update
                discloss, log_dict_disc = self.loss(qloss, inputs, reconstructions,
                                                    optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss

        # "deg2deg": degrade input & reconstruct
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, qloss = self(lres_ims)
            if optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(qloss, lres_ims, reconstructions,
                                                optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            elif optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(qloss,lres_ims, reconstructions,
                                                    optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss

        # "deg2gt": degrade input, but the target is the original HR
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, qloss= self(lres_ims)

            if optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(qloss, hres_ims, reconstructions,
                                                optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return aeloss

            elif optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(qloss, hres_ims, reconstructions, 
                                                    optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
                return discloss

        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        inputs = inputs.to(torch.float)

        if self.ae_mode == 'gt2gt':
            reconstructions, qloss = self(inputs)
            aeloss, log_dict_ae = self.loss(qloss, inputs, reconstructions,
                                            0, self.global_step, last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(qloss, inputs, reconstructions,
                                                1, self.global_step, last_layer=self.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            if len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, qloss= self(lres_ims)
            aeloss, log_dict_ae = self.loss(qloss, lres_ims, reconstructions,
                                            0, self.global_step, last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(qloss, lres_ims, reconstructions, qloss, quant,
                                                1, self.global_step, last_layer=self.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            if len(inputs.shape) == 4:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, qloss, quant = self(lres_ims)
            aeloss, log_dict_ae = self.loss(qloss, hres_ims, reconstructions,
                                            0, self.global_step, last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(qloss, hres_ims, reconstructions,
                                                1, self.global_step, last_layer=self.get_last_layer(), split="val")
            rec_loss = log_dict_ae["val/rec_loss"]

        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        # Typically define self.learning_rate in your config or class
        lr = self.learning_rate
        # The self.loss might have self.discriminator if using a GAN approach
        # or it might just have the vector quantizer. Adjust as needed.
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )

        # If there's a discriminator:
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        Usually used for the adaptive weight in adversarial training (the last conv in the decoder).
        """
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images3D(self, batch, **kwargs):
        """
        Return a dictionary of 3D volumes: "inputs", "reconstructions", "targets"
        for logging or visualization.
        """
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        x = x.float()

        if self.ae_mode == 'gt2gt':
            # no degrade
            xrec, qloss = self(x)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            if len(x.shape) == 4:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=False)
            xrec, qloss = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            if len(x.shape) == 4:
                x = x.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=False)
            xrec, qloss = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims

        return log

    def to_rgb(self, x):
        """
        Optional colorization if your data is label-based. 
        This is similar to what you did in the 2D version.
        """
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        # This is a 2D conv, so x might need shaping, or you adapt to 3D
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
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
    
class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 trainconfig,
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
        self.current_mini_epoch = 0
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.downscaling_factors = trainconfig['downscaling_factors']
        self.ae_mode = trainconfig['ae_mode'] 
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def on_save_checkpoint(self, checkpoint):
        # Save current_mini_epoch so we can resume exactly from the correct mini-epoch
        checkpoint['current_mini_epoch'] = self.current_mini_epoch

    def on_load_checkpoint(self, checkpoint):
        # Restore current_mini_epoch when resuming training
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)

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

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        x = x.to(memory_format=torch.contiguous_format)
        return x

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
        inputs = self.get_input(batch, self.image_key)
        inputs = inputs.to(torch.float)
        if self.ae_mode == 'gt2gt':
            bs, c, h, w = inputs.shape 
            slice_of_interest = c//2 #We are only interested in working with the middle slice. 
            inputs = inputs[:, slice_of_interest, :, :].unsqueeze(1)
            reconstructions, posterior = self(inputs)
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return aeloss

            if optimizer_idx == 1:
                # train the discriminator
                discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")

                self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return discloss
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            #PENDING HERE: 
            reconstructions, posterior = self(lres_ims)
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(lres_ims, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return aeloss

            if optimizer_idx == 1:
                # train the discriminator
                discloss, log_dict_disc = self.loss(lres_ims, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")

                self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return discloss
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            if optimizer_idx == 0:
                # autoencode
                aeloss, log_dict_ae = self.loss(hres_ims, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

                self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return aeloss

            if optimizer_idx == 1:
                # train the discriminator
                discloss, log_dict_disc = self.loss(hres_ims, reconstructions, posterior, optimizer_idx, self.global_step,
                                                    last_layer=self.get_last_layer(), split="train")

                self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
                self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
                return discloss
        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        inputs = inputs.to(torch.float)
        if self.ae_mode == 'gt2gt':
            bs, c, h, w = inputs.shape 
            slice_of_interest = c//2 #We are only interested in working with the middle slice. 
            inputs = inputs[:, slice_of_interest, :, :].unsqueeze(1)
            reconstructions, posterior = self(inputs)
            # autoencode
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

            self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            # autoencode
            aeloss, log_dict_ae = self.loss(lres_ims, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(lres_ims, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

            self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            # autoencode
            aeloss, log_dict_ae = self.loss(hres_ims, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
            discloss, log_dict_disc = self.loss(hres_ims, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="val")

            self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        x = x.to(torch.float)
        if self.ae_mode == 'gt2gt':
            bs, c, h, w = x.shape 
            slice_of_interest = c//2 #We are only interested in working with the middle slice. 
            x = x[:, slice_of_interest, :, :].unsqueeze(1)
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


class AutoencoderKL3D(pl.LightningModule):
    """
    A 3D variant of AutoencoderKL that:
      - Processes inputs of shape [BS, 1, D, H, W]
      - Slices / downscales / upscales for deg2deg or deg2gt tasks
      - Uses an adversarial loss from the configured 'lossconfig'
      - Follows the same logging/naming scheme as the 2D version
      - Employs multiple optimizers in manual optimization (Lightning calls training_step(..., optimizer_idx))
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        trainconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        grad_clip_val=1.0,   # <--- New argument for gradient clipping
    ):
        super().__init__()
        # We are using manual optimization for multiple optimizers
        # (two optimizers -> generator / discriminator)
        self.automatic_optimization = False  
        self.current_mini_epoch = 0
        self.image_key = image_key

        # Build the 3D encoder and decoder from config
        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        # Instantiate the 3D loss/discriminator module
        self.loss = instantiate_from_config(lossconfig)

        # Map from [2*z_channels, D', H', W'] -> [2*embed_dim, D', H', W'] and back
        self.quant_conv = torch.nn.Conv3d(
            2 * ddconfig["z_channels"], 2 * embed_dim, kernel_size=1
        )
        self.post_quant_conv = torch.nn.Conv3d(
            embed_dim, ddconfig["z_channels"], kernel_size=1
        )

        # Additional config
        self.downscaling_factors = trainconfig['downscaling_factors']
        self.ae_mode = trainconfig['ae_mode']
        self.embed_dim = embed_dim

        # Grad accumulation and optimizer logic
        self.current_optimizer_idx = 0
        self.total_batches_epoch = None
        self.is_last_batch = False
        self.backprop_now = False
        self.generator_step_count = 0
        self.discriminator_step_count = 0

        if 'accumulate_grad_batches_g' not in trainconfig.keys() or trainconfig['accumulate_grad_batches_g'] is None:
            print("No gradient accumulation found in trainconfig for generator. Setting to 1.")
            self.accumulate_grad_batches_g = 1
        else:
            print(f'Gradient accumulation for generator set to {trainconfig["accumulate_grad_batches_g"]}')
            self.accumulate_grad_batches_g = trainconfig['accumulate_grad_batches_g']
        
        if 'accumulate_grad_batches_d' not in trainconfig.keys() or trainconfig['accumulate_grad_batches_d'] is None:
            print("No gradient accumulation found in trainconfig for discriminator. Setting to 1.")
            self.accumulate_grad_batches_d = 1
        else:
            print(f'Gradient accumulation for discriminator set to {trainconfig["accumulate_grad_batches_d"]}')
            self.accumulate_grad_batches_d = trainconfig['accumulate_grad_batches_d']

        if colorize_nlabels is not None:
            # For optional colorization (e.g., segmentation)
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Store gradient clipping value
        self.grad_clip_val = grad_clip_val

    def on_save_checkpoint(self, checkpoint):
        # Save current_mini_epoch so we can resume from the correct mini-epoch
        checkpoint['current_mini_epoch'] = self.current_mini_epoch

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading weights from {checkpoint}")
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored 3D checkpoint from {path}")

    # ------------------
    # Forward + encode/decode
    # ------------------
    def encode(self, x):
        """
        Encode a 3D volume [BS, C, D, H, W] -> get posterior q(z|x).
        """
        h = self.encoder(x)                   # [BS, z_channels, D', H', W']
        moments = self.quant_conv(h)          # [BS, 2*embed_dim, D', H', W']
        posterior = DiagonalGaussianDistribution3D(moments)
        return posterior

    def decode(self, z):
        """
        Decode latent z back to 3D volume [BS, C, D, H, W].
        """
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        Standard forward pass: encode -> sample -> decode.
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        """
        Extract the 3D tensor from batch[k].
        If shape is [BS, D, H, W], add a channel dim => [BS, 1, D, H, W].
        """
        x = batch[k]
        if len(x.shape) == 4:  # [BS, D, H, W]
            x = x.unsqueeze(1) # => [BS, 1, D, H, W]
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()
    
    # ------------------
    # Down/Up scale + Blur
    # ------------------
    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        """
        Downscale a 3D tensor [BS, C, D, H, W] -> shape
        """
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        """
        Upscale a 3D tensor back to the desired shape.
        """
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def gaussian_kernel1d(self, sigma, size, device):
        """
        1D Gaussian kernel for 3D blur.
        """
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        """
        3D Gaussian blur via separable conv in each dimension.
        """
        device = img.device
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)

        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, size_h, 1, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, size_w, 1)
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, 1, 1, size_z)

        # Convolve in separate passes
        img = F.conv3d(img, kernel_h, padding=(size_h // 2, 0, 0))
        img = F.conv3d(img, kernel_w, padding=(0, size_w // 2, 0))
        img = F.conv3d(img, kernel_z, padding=(0, 0, size_z // 2))
        return img

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
        """
        Randomly blur a 3D volume with Gaussian of approx sigma ~ uf/2.
        """
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0

        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)

        sigma_z *= (1 + aug_z)
        sigma_h *= (1 + aug_h)
        sigma_w *= (1 + aug_w)

        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
        """
        Create a low-res -> high-res mapping for 3D volumes:
          1) Optionally blur
          2) Downscale
          3) Upscale
        Returns (lres_vol, hres_vol).
        """
        target_vol = img
        bs, c, d, h, w = img.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        # Possibly blur
        if degrade == 'blur':
            img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        # Down/Up scale
        shape = (d // uf_h, h // uf_w, w // uf_z)
        downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
        upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return upscaled_vol, target_vol

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        """
        Randomly pick scaling factors from self.downscaling_factors
        with a small chance for the minimal factor, else uniform among the rest.
        """
        def weighted_choice(options):
            n = len(options)
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)

        if is_train:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        return uf_h, uf_w, uf_z

    # ------------------
    # TRAINING STEP
    # ------------------
    def get_vram_usage(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == 'cpu':
            return {"Alloc": 0, "Resv": 0}
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved  = torch.cuda.memory_reserved(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory
        alloc_GB = mem_allocated / (1024 ** 3)
        resv_GB  = mem_reserved / (1024 ** 3)
        alloc_percent = mem_allocated / total_mem * 100
        resv_percent  = mem_reserved / total_mem * 100
        log = {
            "Alloc": alloc_percent,
            "Resv":  resv_percent,
        }
        return log
    
    def update_optimizer_accum_grad_batches(self, batch_idx):
        """
        Determines whether we should step or accumulate based on the current_optimizer_idx
        and the relevant accumulate_grad_batches setting.
        """
        self.total_batches_epoch = self.trainer.num_training_batches
        if self.current_optimizer_idx == 0:
            # Generator
            if self.total_batches_epoch - batch_idx < self.accumulate_grad_batches_g - 1:
                self.is_last_batch = True
            else:
                self.is_last_batch = False
            if not self.is_last_batch:
                if self.generator_step_count < self.accumulate_grad_batches_g - 1:
                    self.backprop_now = False
                else:
                    self.backprop_now = True
            else:
                # If it's near the end of epoch, we don't want to skip stepping
                if self.generator_step_count < (self.total_batches_epoch - batch_idx - 1):
                    self.backprop_now = False
                else:
                    self.backprop_now = True
        elif self.current_optimizer_idx == 1:
            # Discriminator
            if self.total_batches_epoch - batch_idx < self.accumulate_grad_batches_d - 1:
                self.is_last_batch = True
            else:
                self.is_last_batch = False
            if not self.is_last_batch:
                if self.discriminator_step_count < self.accumulate_grad_batches_d - 1:
                    self.backprop_now = False
                else:
                    self.backprop_now = True
            else:
                if self.discriminator_step_count < (self.total_batches_epoch - batch_idx - 1):
                    self.backprop_now = False
                else:
                    self.backprop_now = True

    def training_step(self, batch, batch_idx):
        """
        Called once per optimizer in manual optimization:
          - If optimizer_idx==0 => Generator (Autoencoder) step
          - If optimizer_idx==1 => Discriminator step
        """
        if batch_idx == 0:
            # Initialize counters at the start of epoch
            self.total_batches_epoch = self.trainer.num_training_batches
            self.current_optimizer_idx = 0
            self.generator_step_count = 0
            self.discriminator_step_count = 0

        opt_ae, opt_disc = self.optimizers()
        inputs = self.get_input(batch, self.image_key).to(torch.float)

        # Switch on ae_mode
        if self.ae_mode == 'gt2gt':
            # No degradation
            # with torch.autocast(device_type="cuda"):
            #     reconstructions, posterior = self(inputs)
            # print(f'lres_ims shape {inputs.shape}')
            # print(f'lres_ims dtype {inputs.type()}')
            reconstructions, posterior = self(inputs)

            if self.current_optimizer_idx == 0:
                # Generator (AE) update
                aeloss, log_dict_ae = self.loss(
                    inputs, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                # Discriminator update
                discloss, log_dict_disc = self.loss(
                    inputs, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            # with torch.autocast(device_type="cuda"):
            # print(f'lres_ims shape {lres_ims.shape}')
            # print(f'lres_ims dtype {lres_ims.type()}')
            reconstructions, posterior = self(lres_ims)

            if self.current_optimizer_idx == 0:
                # Generator
                aeloss, log_dict_ae = self.loss(
                    lres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                # Discriminator
                discloss, log_dict_disc = self.loss(
                    lres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            # with torch.autocast(device_type="cuda"):
            # print(f'lres_ims shape {lres_ims.shape}')
            # print(f'lres_ims dtype {lres_ims.type()}')
            reconstructions, posterior = self(lres_ims)

            if self.current_optimizer_idx == 0:
                # Generator
                aeloss, log_dict_ae = self.loss(
                    hres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                # Discriminator
                discloss, log_dict_disc = self.loss(
                    hres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def validation_step(self, batch, batch_idx):
        """
        Compute AE & Disc losses on validation data; logs like the 2D version.
        We call the loss function for both optimizer_idx=0 & optimizer_idx=1
        just to get the metrics, but we do not do any backward/step here.
        """
        inputs = self.get_input(batch, self.image_key).to(torch.float)

        if self.ae_mode == 'gt2gt':
            # with torch.autocast(device_type="cuda"):
            #     reconstructions, posterior = self(inputs)
            # print(f'lres_ims shape {inputs.shape}')
            # print(f'lres_ims dtype {inputs.type()}')
            
            reconstructions, posterior = self(inputs)
            aeloss, log_dict_ae = self.loss(
                inputs, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            
            self.log("val_rec_loss", log_dict_ae.get("val/rec_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            # with torch.autocast(device_type="cuda"):
            # print(f'lres_ims shape {lres_ims.shape}')
            # print(f'lres_ims dtype {lres_ims.type()}')
            reconstructions, posterior = self(lres_ims)

            aeloss, log_dict_ae = self.loss(
                lres_ims, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                lres_ims, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )

            self.log("val_rec_loss", log_dict_ae.get("val/rec_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            # with torch.autocast(device_type="cuda"):
            # print(f'lres_ims shape {lres_ims.shape}')
            # print(f'lres_ims dtype {lres_ims.type()}')
            reconstructions, posterior = self(lres_ims)

            aeloss, log_dict_ae = self.loss(
                hres_ims, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                hres_ims, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )

            self.log("val_rec_loss", log_dict_ae.get("val/recloss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def test_step(self, batch, batch_idx):
        """
        Typically calls the same logic as validation_step (unless specialized testing).
        """
        return self.validation_step(batch, batch_idx)

    # ------------------
    # CONFIGURE OPTIMIZERS
    # ------------------
    def configure_optimizers(self):
        """
        Return two optimizers => Lightning calls training_step(...) 
        for optimizer_idx = 0, then for optimizer_idx = 1, in each batch
        (manual optimization).
        """
        lr = getattr(self, 'learning_rate', 1e-4)  # if not defined, default
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        Return the final layer for adaptive weight calc, if needed by the loss.
        """
        return self.decoder.conv_out.weight

    # ------------------
    # GRADIENT CLIPPING
    # ------------------
    def on_after_backward(self):
        """
        Called by Lightning after each backward() in manual or automatic optimization.
        We apply gradient clipping here if grad_clip_val is set.
        """
        if self.grad_clip_val is not None and self.grad_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

    # ------------------
    # IMAGE LOGGING
    # ------------------
    @torch.no_grad()
    def log_images3D(self, batch, **kwargs):
        """
        Produce 3D "inputs", "reconstructions", and "targets" for logging.
        """
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device).to(torch.float)

        if self.ae_mode == 'gt2gt':
            # with torch.autocast(device_type="cuda"):
            # print(f'x shape {x.shape}')
            # print(f'x dtype {x.type()}')
            
            xrec, _ = self(x)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            # with torch.autocast(device_type="cuda"):
            # print(f'lres_ims shape {lres_ims.shape}')
            # print(f'lres_ims dtype {lres_ims.type()}')
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            # with torch.autocast(device_type="cuda"):
            # print(f'lres_ims shape {lres_ims.shape}')
            # print(f'lres_ims dtype {lres_ims.type()}')
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims

        return log

    def to_rgb(self, x):
        """
        Converts volumes to RGB if needed. Typically for segmentation or multi-channel volumes.
        """
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

class AutoencoderKL3DV2(pl.LightningModule):
    """
    A 3D variant of AutoencoderKL that:
      - Processes inputs of shape [BS, 1, D, H, W]
      - Slices / downscales / upscales for deg2deg or deg2gt tasks
      - Uses an adversarial loss from the configured 'lossconfig'
      - Follows the same logging/naming scheme as the 2D version
      - Employs multiple optimizers in manual optimization
      
    New in V2:
      - Adds a parameter 'load_pretrained_weights_mode' which can be set to:
          "both"      -> load both generator and discriminator weights (default)
          "generator" -> load only the generator weights.
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        trainconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        grad_clip_val=1.0,   # New argument for gradient clipping.
        load_pretrained_weights_mode="generator"  # New parameter: "both" or "generator"
    ):
        super().__init__()
        self.automatic_optimization = False  
        self.current_mini_epoch = 0
        self.image_key = image_key
        self.load_pretrained_weights_mode = load_pretrained_weights_mode

        # Build the 3D encoder and decoder from config.
        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        # Instantiate the 3D loss/discriminator module.
        self.loss = instantiate_from_config(lossconfig)

        # Map from [2*z_channels, D', H', W'] -> [2*embed_dim, D', H', W'] and back.
        self.quant_conv = torch.nn.Conv3d(
            2 * ddconfig["z_channels"], 2 * embed_dim, kernel_size=1
        )
        self.post_quant_conv = torch.nn.Conv3d(
            embed_dim, ddconfig["z_channels"], kernel_size=1
        )

        # Additional config.
        self.downscaling_factors = trainconfig['downscaling_factors']
        self.ae_mode = trainconfig['ae_mode']
        self.embed_dim = embed_dim

        # Grad accumulation and optimizer logic.
        self.current_optimizer_idx = 0
        self.total_batches_epoch = None
        self.is_last_batch = False
        self.backprop_now = False
        self.generator_step_count = 0
        self.discriminator_step_count = 0

        if 'accumulate_grad_batches_g' not in trainconfig.keys() or trainconfig['accumulate_grad_batches_g'] is None:
            print("No gradient accumulation found in trainconfig for generator. Setting to 1.")
            self.accumulate_grad_batches_g = 1
        else:
            print(f'Gradient accumulation for generator set to {trainconfig["accumulate_grad_batches_g"]}')
            self.accumulate_grad_batches_g = trainconfig['accumulate_grad_batches_g']
        
        if 'accumulate_grad_batches_d' not in trainconfig.keys() or trainconfig['accumulate_grad_batches_d'] is None:
            print("No gradient accumulation found in trainconfig for discriminator. Setting to 1.")
            self.accumulate_grad_batches_d = 1
        else:
            print(f'Gradient accumulation for discriminator set to {trainconfig["accumulate_grad_batches_d"]}')
            self.accumulate_grad_batches_d = trainconfig['accumulate_grad_batches_d']

        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        # If a checkpoint path is provided, load weights.
        # If loading only generator weights, add discriminator keys to ignore_keys.
        if ckpt_path is not None:
            if self.load_pretrained_weights_mode == "generator":
                ignore_keys = ignore_keys + ["loss.discriminator"]
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Store gradient clipping value.
        self.grad_clip_val = grad_clip_val

    def on_save_checkpoint(self, checkpoint):
        checkpoint['current_mini_epoch'] = self.current_mini_epoch

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading weights from {checkpoint}")
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict as per ignore_keys.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored 3D checkpoint from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution3D(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:  # [BS, D, H, W]
            x = x.unsqueeze(1)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()
    
    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def gaussian_kernel1d(self, sigma, size, device):
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        device = img.device
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)

        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, size_h, 1, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, size_w, 1)
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, 1, 1, size_z)

        img = F.conv3d(img, kernel_h, padding=(size_h // 2, 0, 0))
        img = F.conv3d(img, kernel_w, padding=(0, size_w // 2, 0))
        img = F.conv3d(img, kernel_z, padding=(0, 0, size_z // 2))
        return img

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0

        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)

        sigma_z *= (1 + aug_z)
        sigma_h *= (1 + aug_h)
        sigma_w *= (1 + aug_w)

        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
        target_vol = img
        bs, c, d, h, w = img.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        if degrade == 'blur':
            img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        shape = (d // uf_h, h // uf_w, w // uf_z)
        downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
        upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return upscaled_vol, target_vol

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        def weighted_choice(options):
            n = len(options)
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)

        if is_train:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        return uf_h, uf_w, uf_z

    def get_vram_usage(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == 'cpu':
            return {"Alloc": 0, "Resv": 0}
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved  = torch.cuda.memory_reserved(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory
        alloc_percent = mem_allocated / total_mem * 100
        resv_percent  = mem_reserved / total_mem * 100
        log = {
            "Alloc": alloc_percent,
            "Resv":  resv_percent,
        }
        return log

    def update_optimizer_accum_grad_batches(self, batch_idx):
        self.total_batches_epoch = self.trainer.num_training_batches
        if self.current_optimizer_idx == 0:
            if self.total_batches_epoch - batch_idx < self.accumulate_grad_batches_g - 1:
                self.is_last_batch = True
            else:
                self.is_last_batch = False
            if not self.is_last_batch:
                if self.generator_step_count < self.accumulate_grad_batches_g - 1:
                    self.backprop_now = False
                else:
                    self.backprop_now = True
            else:
                if self.generator_step_count < (self.total_batches_epoch - batch_idx - 1):
                    self.backprop_now = False
                else:
                    self.backprop_now = True
        elif self.current_optimizer_idx == 1:
            if self.total_batches_epoch - batch_idx < self.accumulate_grad_batches_d - 1:
                self.is_last_batch = True
            else:
                self.is_last_batch = False
            if not self.is_last_batch:
                if self.discriminator_step_count < self.accumulate_grad_batches_d - 1:
                    self.backprop_now = False
                else:
                    self.backprop_now = True
            else:
                if self.discriminator_step_count < (self.total_batches_epoch - batch_idx - 1):
                    self.backprop_now = False
                else:
                    self.backprop_now = True

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.total_batches_epoch = self.trainer.num_training_batches
            self.current_optimizer_idx = 0
            self.generator_step_count = 0
            self.discriminator_step_count = 0

        opt_ae, opt_disc = self.optimizers()
        inputs = self.get_input(batch, self.image_key).to(torch.float)

        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)

            if self.current_optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(
                    inputs, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(
                    inputs, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            if self.current_optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(
                    lres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(
                    lres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            if self.current_optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(
                    hres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(
                    hres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key).to(torch.float)

        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)
            aeloss, log_dict_ae = self.loss(
                inputs, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            
            self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            aeloss, log_dict_ae = self.loss(
                lres_ims, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                lres_ims, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )

            self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            aeloss, log_dict_ae = self.loss(
                hres_ims, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                hres_ims, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )

            self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        lr = getattr(self, 'learning_rate', 1e-4)
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(),
            lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_after_backward(self):
        if self.grad_clip_val is not None and self.grad_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

    @torch.no_grad()
    def log_images3D(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device).to(torch.float)

        if self.ae_mode == 'gt2gt':
            xrec, _ = self(x)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

class AutoencoderKL3DV3(pl.LightningModule):
    """
    A 3D variant of AutoencoderKL that:
      - Processes inputs of shape [BS, 1, D, H, W]
      - Slices / downscales / upscales for deg2deg or deg2gt tasks
      - Uses an adversarial loss from the configured 'lossconfig'
      - Follows the same logging/naming scheme as the 2D version
      - Employs multiple optimizers in manual optimization
      
    New in V2:
      - Adds a parameter 'load_pretrained_weights_mode' which can be set to:
          "both"      -> load both generator and discriminator weights (default)
          "generator" -> load only the generator weights.
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        trainconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        grad_clip_val=1.0,   # New argument for gradient clipping.
        load_pretrained_weights_mode="both"  # New parameter: "both" or "generator"
    ):
        super().__init__()
        self.automatic_optimization = False  
        self.current_mini_epoch = 0
        self.image_key = image_key
        self.load_pretrained_weights_mode = load_pretrained_weights_mode

        # Build the 3D encoder and decoder from config.
        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        # Instantiate the 3D loss/discriminator module.
        self.loss = instantiate_from_config(lossconfig)

        # Map from [2*z_channels, D', H', W'] -> [2*embed_dim, D', H', W'] and back.
        self.quant_conv = torch.nn.Conv3d(
            2 * ddconfig["z_channels"], 2 * embed_dim, kernel_size=1
        )
        self.post_quant_conv = torch.nn.Conv3d(
            embed_dim, ddconfig["z_channels"], kernel_size=1
        )

        # Additional config.
        self.downscaling_factors = trainconfig['downscaling_factors']
        self.ae_mode = trainconfig['ae_mode']
        self.embed_dim = embed_dim
        
        # Grad accumulation and optimizer logic.
        self.current_optimizer_idx = 0
        self.total_batches_epoch = None
        self.is_last_batch = False
        self.backprop_now = False
        self.generator_step_count = 0
        self.discriminator_step_count = 0

        if 'accumulate_grad_batches_g' not in trainconfig.keys() or trainconfig['accumulate_grad_batches_g'] is None:
            print("No gradient accumulation found in trainconfig for generator. Setting to 1.")
            self.accumulate_grad_batches_g = 1
        else:
            print(f'Gradient accumulation for generator set to {trainconfig["accumulate_grad_batches_g"]}')
            self.accumulate_grad_batches_g = trainconfig['accumulate_grad_batches_g']
        
        if 'accumulate_grad_batches_d' not in trainconfig.keys() or trainconfig['accumulate_grad_batches_d'] is None:
            print("No gradient accumulation found in trainconfig for discriminator. Setting to 1.")
            self.accumulate_grad_batches_d = 1
        else:
            print(f'Gradient accumulation for discriminator set to {trainconfig["accumulate_grad_batches_d"]}')
            self.accumulate_grad_batches_d = trainconfig['accumulate_grad_batches_d']


        # self.lr_ratio_gtod = trainconfig['lr_ratio_gtod']
        if 'lr_ratio_gtod' not in trainconfig.keys() or trainconfig['lr_ratio_gtod'] is None:
            print("No lr_ratio_gtod (Generator_LR/Discriminator_LR). Setting to default 10")
            self.lr_ratio_gtod = 10
        else:
            print(f'Setting lr_ratio_gtod to the found value of {trainconfig["lr_ratio_gtod"]}')
            self.lr_ratio_gtod = trainconfig['lr_ratio_gtod']

        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        # If a checkpoint path is provided, load weights.
        # If loading only generator weights, add discriminator keys to ignore_keys.
        if ckpt_path is not None:
            if self.load_pretrained_weights_mode == "generator":
                ignore_keys = ignore_keys + ["loss.discriminator"]
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # Store gradient clipping value.
        self.grad_clip_val = grad_clip_val

    def on_save_checkpoint(self, checkpoint):
        checkpoint['current_mini_epoch'] = self.current_mini_epoch

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading weights from {checkpoint}")
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict as per ignore_keys.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored 3D checkpoint from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution3D(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:  # [BS, D, H, W]
            x = x.unsqueeze(1)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()
    
    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def gaussian_kernel1d(self, sigma, size, device):
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        device = img.device
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)

        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, size_h, 1, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, size_w, 1)
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, 1, 1, size_z)

        img = F.conv3d(img, kernel_h, padding=(size_h // 2, 0, 0))
        img = F.conv3d(img, kernel_w, padding=(0, size_w // 2, 0))
        img = F.conv3d(img, kernel_z, padding=(0, 0, size_z // 2))
        return img

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0

        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)

        sigma_z *= (1 + aug_z)
        sigma_h *= (1 + aug_h)
        sigma_w *= (1 + aug_w)

        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
        target_vol = img
        bs, c, d, h, w = img.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        if degrade == 'blur':
            img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        shape = (d // uf_h, h // uf_w, w // uf_z)
        downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
        upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return upscaled_vol, target_vol

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        def weighted_choice(options):
            n = len(options)
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)

        if is_train:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else:
            if coupled_ufs:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = uf_h
                uf_z = uf_h
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        return uf_h, uf_w, uf_z

    def get_vram_usage(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == 'cpu':
            return {"Alloc": 0, "Resv": 0}
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved  = torch.cuda.memory_reserved(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory
        alloc_percent = mem_allocated / total_mem * 100
        resv_percent  = mem_reserved / total_mem * 100
        log = {
            "Alloc": alloc_percent,
            "Resv":  resv_percent,
        }
        return log

    def update_optimizer_accum_grad_batches(self, batch_idx):
        self.total_batches_epoch = self.trainer.num_training_batches
        if self.current_optimizer_idx == 0:
            if self.total_batches_epoch - batch_idx < self.accumulate_grad_batches_g - 1:
                self.is_last_batch = True
            else:
                self.is_last_batch = False
            if not self.is_last_batch:
                if self.generator_step_count < self.accumulate_grad_batches_g - 1:
                    self.backprop_now = False
                else:
                    self.backprop_now = True
            else:
                if self.generator_step_count < (self.total_batches_epoch - batch_idx - 1):
                    self.backprop_now = False
                else:
                    self.backprop_now = True
        elif self.current_optimizer_idx == 1:
            if self.total_batches_epoch - batch_idx < self.accumulate_grad_batches_d - 1:
                self.is_last_batch = True
            else:
                self.is_last_batch = False
            if not self.is_last_batch:
                if self.discriminator_step_count < self.accumulate_grad_batches_d - 1:
                    self.backprop_now = False
                else:
                    self.backprop_now = True
            else:
                if self.discriminator_step_count < (self.total_batches_epoch - batch_idx - 1):
                    self.backprop_now = False
                else:
                    self.backprop_now = True

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.total_batches_epoch = self.trainer.num_training_batches
            self.current_optimizer_idx = 0
            self.generator_step_count = 0
            self.discriminator_step_count = 0

        opt_ae, opt_disc = self.optimizers()
        inputs = self.get_input(batch, self.image_key).to(torch.float)

        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)

            if self.current_optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(
                    inputs, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(
                    inputs, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            if self.current_optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(
                    lres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(
                    lres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            if self.current_optimizer_idx == 0:
                aeloss, log_dict_ae = self.loss(
                    hres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                ae_loss = aeloss / self.accumulate_grad_batches_g
                self.manual_backward(ae_loss)
                self.generator_step_count += 1

                if self.backprop_now:
                    opt_ae.step()
                    opt_ae.zero_grad()
                    self.generator_step_count = 0
                    self.current_optimizer_idx = 1

                return aeloss

            elif self.current_optimizer_idx == 1:
                discloss, log_dict_disc = self.loss(
                    hres_ims, reconstructions, posterior,
                    self.current_optimizer_idx, self.global_step,
                    last_layer=self.get_last_layer(),
                    split="train"
                )
                vram_log = self.get_vram_usage()
                self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
                self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

                self.update_optimizer_accum_grad_batches(batch_idx)
                discloss = discloss / self.accumulate_grad_batches_d
                self.manual_backward(discloss)
                self.discriminator_step_count += 1

                if self.backprop_now:
                    opt_disc.step()
                    opt_disc.zero_grad()
                    self.discriminator_step_count = 0
                    self.current_optimizer_idx = 0

                return discloss

        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key).to(torch.float)

        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)
            aeloss, log_dict_ae = self.loss(
                inputs, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                inputs, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            
            self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            aeloss, log_dict_ae = self.loss(
                lres_ims, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                lres_ims, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )

            self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)

            aeloss, log_dict_ae = self.loss(
                hres_ims, reconstructions, posterior, 0,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )
            discloss, log_dict_disc = self.loss(
                hres_ims, reconstructions, posterior, 1,
                self.global_step, last_layer=self.get_last_layer(),
                split="val"
            )

            self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log})
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return {**log_dict_ae, **log_dict_disc}
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        lr = getattr(self, 'learning_rate', 1e-4)
        opt_ae = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.post_quant_conv.parameters()),
            lr=lr, betas=(0.5, 0.9)
        )
        opt_disc = torch.optim.AdamW(
            self.loss.discriminator.parameters(),
            lr=lr/self.lr_ratio_gtod, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_after_backward(self):
        if self.grad_clip_val is not None and self.grad_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

    @torch.no_grad()
    def log_images3D(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device).to(torch.float)

        if self.ae_mode == 'gt2gt':
            xrec, _ = self(x)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=True)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims

        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x

class AutoencoderKL3DFiLM_BiCond(pl.LightningModule): # New class name
    """
    A 3D AutoencoderKL variant that uses:
    - FiLM-conditioned Encoder3D (on SOURCE voxel size)
    - FiLM-conditioned Decoder3D (on TARGET voxel size)
    - Handles calculation of source voxel size from target and UFs.
    - Handles selective loading of weights for fine-tuning.
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        trainconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        target_cond_key="zoom", # Key for TARGET voxel size in batch
        # --- Encoder FiLM Params ---
        enc_cond_input_dim=3,
        enc_cond_embed_dim=256, # Can be different from decoder
        # --- Decoder FiLM Params ---
        dec_cond_input_dim=3, # Usually same input dim (3)
        dec_cond_embed_dim=256, # Can be different from encoder
        # --- Loading Mode ---
        load_pretrained_weights_mode="both", # "both", "generator", "encoder_only", "decoder_only" ? Add flexibility later if needed
        # --- Other Params ---
        monitor="val_rec_loss", # Default monitor
        logvar_init=0.0,      # Parameter for NLLLoss if used in lossconfig
        grad_clip_val=1.0,
        **kwargs # Catch any other potential args like colorize_nlabels etc.
    ):
        super().__init__()
        self.automatic_optimization = False
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init) # Example if loss uses it directly

        # Store config snippets
        self.ddconfig = ddconfig
        self.lossconfig = lossconfig
        self.trainconfig = trainconfig
        self.embed_dim = embed_dim
        self.image_key = image_key
        self.target_cond_key = target_cond_key # Key for TARGET voxel size
        self.grad_clip_val = grad_clip_val
        self.load_pretrained_weights_mode = load_pretrained_weights_mode
        self.monitor = monitor

        # --- Build Models ---
        # Encoder with FiLM
        encoder_params = self.ddconfig.copy()
        encoder_params['cond_input_dim'] = enc_cond_input_dim
        encoder_params['cond_embed_dim'] = enc_cond_embed_dim
        self.encoder = Encoder3DFiLM(**encoder_params)

        # Decoder with FiLM
        decoder_params = self.ddconfig.copy()
        decoder_params['cond_input_dim'] = dec_cond_input_dim
        decoder_params['cond_embed_dim'] = dec_cond_embed_dim
        self.decoder = Decoder3DFiLM(**decoder_params)
        # --- End Build Models ---

        # Loss Module (Discriminator)
        self.loss = instantiate_from_config(self.lossconfig)

        # Quant/Post-Quant Convs
        self.quant_conv = nn.Conv3d(2 * self.ddconfig["z_channels"], 2 * self.embed_dim, 1)
        self.post_quant_conv = nn.Conv3d(self.embed_dim, self.ddconfig["z_channels"], 1)

        # Training state variables
        self.current_optimizer_idx = 0
        self.generator_step_count = 0
        self.discriminator_step_count = 0
        # Parse trainconfig params safely
        self.downscaling_factors = self.trainconfig.get('downscaling_factors', [1, 2, 3])
        self.ae_mode = self.trainconfig.get('ae_mode', 'deg2gt')
        self.accumulate_grad_batches_g = self.trainconfig.get('accumulate_grad_batches_g', 1)
        self.accumulate_grad_batches_d = self.trainconfig.get('accumulate_grad_batches_d', 1)
        self.lr_ratio_gtod = self.trainconfig.get('lr_ratio_gtod', 1.0) # Default to 1 if not specified

        # --- Checkpoint Loading ---
        if ckpt_path is not None:
            print(f"AutoencoderKL3DFiLM_BiCond: Initializing from checkpoint: {ckpt_path}")
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys,
                                load_mode=self.load_pretrained_weights_mode)
        else:
            print("AutoencoderKL3DFiLM_BiCond: No checkpoint path provided, initializing from scratch.")

    # Override init_from_ckpt for BiCond model
    def init_from_ckpt(self, path, ignore_keys=list(), load_mode="both"):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        print(f"AutoencoderKL3DFiLM_BiCond: Restoring ckpt from {path} with load_mode='{load_mode}'.")

        final_ignore_keys = list(ignore_keys)

        # Define prefixes for components
        enc_prefix = "encoder."
        dec_prefix = "decoder."
        disc_prefix = "loss.discriminator."
        quant_prefix = "quant_conv."
        post_quant_prefix = "post_quant_conv."
        # FiLM specific key identifiers (adjust if your layer names differ)
        film_key_parts = ["cond_proj", "cond_embedder"]

        keys_to_ignore = set(final_ignore_keys)

        if load_mode == "generator":
            print("BiCond: Generator-only load mode. Ignoring Disc & ALL FiLM layers.")
            # Ignore discriminator
            for k in keys:
                if k.startswith(disc_prefix): keys_to_ignore.add(k)
            # Ignore ALL FiLM layers (Encoder + Decoder) from checkpoint if they exist
            for k in keys:
                if any(part in k for part in film_key_parts):
                     keys_to_ignore.add(k)
                     # print(f"BiCond: Ignoring potential FiLM key from ckpt: {k}") # Verbose

        elif load_mode == "both":
            print("BiCond: Loading all compatible weights (Gen+Disc). FiLM layers initialized randomly if missing in ckpt.")
            # If loading OLD ckpt into NEW BiCond model: FiLM layers are missing, `strict=False` handles it.
            # If loading NEW BiCond ckpt into NEW BiCond model: Load everything.
            pass # No additional ignores needed beyond user list

        # Add more modes like "encoder_only" if needed later
        else:
            raise ValueError(f"Unknown load_pretrained_weights_mode: {load_mode}")

        # Remove ignored keys from state dict
        original_key_count = len(sd)
        sd = {k: v for k, v in sd.items() if k not in keys_to_ignore}
        deleted_count = original_key_count - len(sd)
        print(f"BiCond: Deleted {deleted_count} keys based on load_mode and ignore_keys.")

        # Load with strict=False
        load_result = self.load_state_dict(sd, strict=False)
        print(f"BiCond: Checkpoint loading results:")
        print(f"  Missing keys: {load_result.missing_keys}") # Should include FiLM keys if loading old ckpt or mode=generator
        print(f"  Unexpected keys: {load_result.unexpected_keys}") # Should be empty

        print(f"BiCond: Successfully attempted weight restoration from {path}.")


    # --- Encode/Decode Methods modified for conditioning ---
    def encode(self, x, source_cond_input=None):
        # Pass source conditioning to Encoder3DFiLM
        h = self.encoder(x, cond_input=source_cond_input)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution3D(moments)
        return posterior

    def decode(self, z, target_cond_input=None):
        z = self.post_quant_conv(z)
        # Pass target conditioning to Decoder3DFiLM
        dec = self.decoder(z, cond_input=target_cond_input)
        return dec

    # --- Forward method needs adaptation (or call encode/decode directly) ---
    # Option 1: Modify forward (cleaner if possible)
    def forward(self, input, source_cond_input=None, target_cond_input=None, sample_posterior=True):
        posterior = self.encode(input, source_cond_input=source_cond_input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z, target_cond_input=target_cond_input)
        return dec, posterior

    # --- Helper to get input data ---
    def get_input(self, batch, k):
        x = batch[k]
        if isinstance(x, list): # Handle cases like 'zoom' which might be list
             x = torch.tensor(x) # Convert list to tensor if needed
        if len(x.shape) == 4: x = x.unsqueeze(1) # Add channel dim if needed
        x = x.to(memory_format=torch.contiguous_format).float() # Ensure float
        return x

    # --- Helper to calculate source voxel size ---
    def calculate_source_voxel_size(self, target_voxel_size, ufs):
        """Calculates source voxel size before upsampling back."""
        # Ensure target_voxel_size is a tensor on the correct device
        if not isinstance(target_voxel_size, torch.Tensor):
            # Assuming target_voxel_size is [bs, 3] list/numpy array
            target_voxel_size = torch.tensor(target_voxel_size, dtype=torch.float32, device=self.device)
        elif target_voxel_size.device != self.device:
            target_voxel_size = target_voxel_size.to(self.device)

        # Ensure ufs is a tensor [3] on the correct device
        ufs_tensor = torch.tensor(ufs, dtype=torch.float32, device=self.device) # Shape [3]
        ufs_tensor = ufs_tensor.unsqueeze(0) # Shape [1, 3] for broadcasting

        # Element-wise multiplication: [bs, 3] * [1, 3] -> [bs, 3]
        source_voxel_size = target_voxel_size * ufs_tensor
        return source_voxel_size

    # --- Training Step Modified ---
    def training_step(self, batch, batch_idx):
        # 1. Get Target Voxel Size (Condition for Decoder)
        target_voxel_size = self.get_input(batch, self.target_cond_key) # Shape [bs, 3]

        # 2. Get High-Res Input Image
        inputs = self.get_input(batch, self.image_key) # Shape [bs, c, d, h, w]

        # 3. Determine Degradation & Calculate Source Voxel Size (Condition for Encoder)
        if self.ae_mode in ['deg2deg', 'deg2gt']:
            # Choose random upsampling factors
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            ufs = [uf_h, uf_w, uf_z] # Store as list/tuple

            # Calculate source voxel size BEFORE generating degraded image
            # Note: This represents the voxel size of the *downsampled* volume
            source_voxel_size = self.calculate_source_voxel_size(target_voxel_size, ufs)

            # Prepare degraded input (lres) and target (hres or lres)
            lres_ims, targets = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)

            # Call forward with both conditions
            reconstructions, posterior = self(lres_ims,
                                              source_cond_input=source_voxel_size,
                                              target_cond_input=target_voxel_size)

        elif self.ae_mode == 'gt2gt':
            # No degradation, source = target
            source_voxel_size = target_voxel_size
            lres_ims = inputs # Input is the original GT
            targets = inputs # Target is the original GT
            reconstructions, posterior = self(inputs,
                                              source_cond_input=source_voxel_size,
                                              target_cond_input=target_voxel_size)
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")


        # 4. Loss Calculation and Optimizer Steps (Identical Logic to AutoencoderKL3DFiLM)
        opt_ae, opt_disc = self.optimizers() # Get optimizers

        # --- Generator Update ---
        if self.current_optimizer_idx == 0:
            aeloss, log_dict_ae = self.loss(targets, reconstructions, posterior, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            if self.global_rank == 0 and batch_idx % 100 == 0: # Print occasionally on rank 0
                print(f"\n--- Debugging log_dict_ae (Step {self.global_step}) ---")
                for key, value in log_dict_ae.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  Key: {key}, Device: {value.device}, Shape: {value.shape}, Dtype: {value.dtype}")
                    else:
                        print(f"  Key: {key}, Type: {type(value)}, Value: {value}")
                print("--- End Debug ---")
            self.log_dict(log_dict_ae, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            # Grad Accumulation
            self.update_optimizer_accum_grad_batches(batch_idx) # Needs access to self.trainer
            ae_loss_scaled = aeloss / self.accumulate_grad_batches_g
            self.manual_backward(ae_loss_scaled) # Backward scaled loss
            self.generator_step_count += 1

            # Optimizer Step
            if self.backprop_now:
                # self.on_before_optimizer_step(opt_ae, 0) # Clip grads
                opt_ae.step()
                opt_ae.zero_grad()
                self.generator_step_count = 0
                self.current_optimizer_idx = 1 # Switch to discriminator

            return aeloss # Return unscaled loss for logging

        # --- Discriminator Update ---
        elif self.current_optimizer_idx == 1:
            discloss, log_dict_disc = self.loss(targets, reconstructions, posterior, 1, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            # Grad Accumulation
            self.update_optimizer_accum_grad_batches(batch_idx) # Needs access to self.trainer
            disc_loss_scaled = discloss / self.accumulate_grad_batches_d
            self.manual_backward(disc_loss_scaled) # Backward scaled loss
            self.discriminator_step_count += 1

            # Optimizer Step
            if self.backprop_now:
                # self.on_before_optimizer_step(opt_disc, 1) # Clip grads
                opt_disc.step()
                opt_disc.zero_grad()
                self.discriminator_step_count = 0
                self.current_optimizer_idx = 0 # Switch back to generator

            return discloss # Return unscaled loss for logging


    # --- Validation Step Modified ---
    def validation_step(self, batch, batch_idx):
        # Similar logic as training_step, but no backward pass/optimizer steps
        target_voxel_size = self.get_input(batch, self.target_cond_key)
        inputs = self.get_input(batch, self.image_key)

        if self.ae_mode in ['deg2deg', 'deg2gt']:
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False) # is_train=False
            ufs = [uf_h, uf_w, uf_z]
            source_voxel_size = self.calculate_source_voxel_size(target_voxel_size, ufs)
            lres_ims, targets = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, posterior = self(lres_ims, source_cond_input=source_voxel_size,
                                              target_cond_input=target_voxel_size)
        elif self.ae_mode == 'gt2gt':
            source_voxel_size = target_voxel_size
            targets = inputs
            reconstructions, posterior = self(inputs, source_cond_input=source_voxel_size,
                                              target_cond_input=target_voxel_size)
        else: raise NotImplementedError()

        # Calculate losses
        aeloss, log_dict_ae = self.loss(targets, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(targets, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        # Log metrics
        self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss), sync_dist=True) # Use same key for checkpointing
        self.log_dict(log_dict_ae, sync_dist=True)
        self.log_dict(log_dict_disc, sync_dist=True)
        return self.log_dict # Return logged dict


    # --- Need to reimplement shared methods or ensure inheritance works ---
    # Make sure methods like get_batch_ufs_and_noise_std, prepare_input_target_torch,
    # update_optimizer_accum_grad_batches, configure_optimizers, get_last_layer,
    # on_before_optimizer_step, log_images3D are either inherited correctly
    # from AutoencoderKL3DV3/FiLM or reimplemented here.

    # Example: Ensure configure_optimizers is present
    def configure_optimizers(self):
        lr = self.learning_rate # learning_rate should be set by main script
        opt_ae_params = list(self.encoder.parameters()) + \
                        list(self.decoder.parameters()) + \
                        list(self.quant_conv.parameters()) + \
                        list(self.post_quant_conv.parameters())
        opt_ae = torch.optim.AdamW(opt_ae_params, lr=lr, betas=(0.5, 0.9))

        opt_disc_params = self.loss.discriminator.parameters()
        disc_lr = lr / self.lr_ratio_gtod # Use lr_ratio_gtod
        opt_disc = torch.optim.AdamW(opt_disc_params, lr=disc_lr, betas=(0.5, 0.9))

        print(f"BiCond: Setting AE LR: {lr:.2e}, Disc LR: {disc_lr:.2e} (ratio G/D: {self.lr_ratio_gtod})")
        return [opt_ae, opt_disc], []

    # Example: Ensure get_last_layer is present
    def get_last_layer(self):
        # Ensure decoder has conv_out attribute
        try:
             return self.decoder.conv_out.weight
        except AttributeError:
             print("Warning: Decoder does not have conv_out attribute for adaptive weight calculation.")
             # Find the last parameter of the decoder as a fallback
             last_param = None
             for param in reversed(list(self.decoder.parameters())):
                  if param.requires_grad:
                       last_param = param
                       break
             if last_param is not None:
                  print("Warning: Using last parameter of decoder as fallback for last_layer.")
                  return last_param
             else:
                  # This should not happen if decoder has parameters
                  # Fallback further to a non-learnable tensor? Or raise error?
                  # Let's return a dummy tensor and rely on d_weight=0 if this fails
                  print("Critical Warning: Cannot find last layer for adaptive weight. Using dummy.")
                  return torch.zeros(1) # Or handle in loss calculation


    # --- Need implementations for methods used in training_step ---
    # These were likely in AutoencoderKL3DV3, ensure they are accessible here
    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        # --- Replicate or inherit this method ---
        # Example implementation:
        def weighted_choice(options):
            n = len(options)
            # Example weighting: favour factor 1 slightly more maybe? Adjust as needed.
            # probabilities = [0.4] + [0.6 / (n - 1)] * (n - 1) if n > 1 else [1.0]
            # Original weighting from AutoencoderKL3DV3 seemed to be:
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1) if n > 1 else [1.0]
            return np.random.choice(options, p=probabilities)

        if coupled_ufs:
            uf_h = weighted_choice(self.downscaling_factors)
            uf_w = uf_h
            uf_z = uf_h
        else:
            uf_h = weighted_choice(self.downscaling_factors)
            uf_w = weighted_choice(self.downscaling_factors)
            uf_z = weighted_choice(self.downscaling_factors)
        return uf_h, uf_w, uf_z

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
         # --- Replicate or inherit this method ---
         # Example implementation (assuming blur_tensor, downscale, upscale exist):
         target_vol = img
         bs, c, d, h, w = img.shape
         interpolation_method = 'trilinear'
         # Example: Always blur for deg modes? Or make it random?
         degrade = 'blur' # Or random.choice(['blur', 'noblur'])

         if degrade == 'blur':
              # Ensure self.blur_tensor exists and is compatible
              img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

         # Calculate downsampled shape correctly (integer division)
         shape = (d // int(uf_z), h // int(uf_h), w // int(uf_w)) # Use factors in Z, H, W order
         # Ensure shape dimensions are at least 1
         shape = tuple(max(1, s) for s in shape)

         downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
         # Upsample back to original D, H, W
         upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear', align_corners=True) # align_corners=True for consistency?

         if self.ae_mode == 'deg2deg':
              return upscaled_vol, upscaled_vol # Target is also degraded
         else: # deg2gt
              return upscaled_vol, target_vol

    # --- Need blur_tensor, downscale, upscale ---
    # Add these methods if they are not inherited
    def gaussian_kernel1d(self, sigma, size, device):
         # Simple 1D Gaussian kernel generation
         x_cord = torch.arange(size, device=device)
         x_grid = x_cord - size // 2
         g = torch.exp(-x_grid**2 / (2 * sigma**2))
         return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
         # Simple separable 3D Gaussian blur
         device = img.device
         # Determine kernel sizes (odd number)
         k_size_z = int(2 * np.ceil(2 * sigma_z) + 1) if sigma_z > 0 else 1
         k_size_h = int(2 * np.ceil(2 * sigma_h) + 1) if sigma_h > 0 else 1
         k_size_w = int(2 * np.ceil(2 * sigma_w) + 1) if sigma_w > 0 else 1

         # Generate 1D kernels
         kernel_z = self.gaussian_kernel1d(sigma_z, k_size_z, device).view(1, 1, k_size_z, 1, 1) if sigma_z > 0 else None
         kernel_h = self.gaussian_kernel1d(sigma_h, k_size_h, device).view(1, 1, 1, k_size_h, 1) if sigma_h > 0 else None
         kernel_w = self.gaussian_kernel1d(sigma_w, k_size_w, device).view(1, 1, 1, 1, k_size_w) if sigma_w > 0 else None

         # Apply convolution if kernel exists
         channels = img.shape[1]
         if kernel_z is not None:
              kernel_z = kernel_z.repeat(channels, 1, 1, 1, 1) # Repeat for group convolution
              img = F.conv3d(img, kernel_z, padding=(k_size_z // 2, 0, 0), groups=channels)
         if kernel_h is not None:
              kernel_h = kernel_h.repeat(channels, 1, 1, 1, 1)
              img = F.conv3d(img, kernel_h, padding=(0, k_size_h // 2, 0), groups=channels)
         if kernel_w is not None:
              kernel_w = kernel_w.repeat(channels, 1, 1, 1, 1)
              img = F.conv3d(img, kernel_w, padding=(0, 0, k_size_w // 2), groups=channels)
         return img

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
         # Simple blur based on upsampling factors
         # Sigma chosen as UF/2, potentially augmented
         sigma_z = (uf_z / 2.0) * (1 + np.random.uniform(-p_aug_sigma, p_aug_sigma)) if uf_z > 1 else 0
         sigma_h = (uf_h / 2.0) * (1 + np.random.uniform(-p_aug_sigma, p_aug_sigma)) if uf_h > 1 else 0
         sigma_w = (uf_w / 2.0) * (1 + np.random.uniform(-p_aug_sigma, p_aug_sigma)) if uf_w > 1 else 0

         # Apply blur only if sigma > 0
         if sigma_z > 0 or sigma_h > 0 or sigma_w > 0:
              return self.gaussian_blur(img, max(0, sigma_z), max(0, sigma_h), max(0, sigma_w))
         else:
              return img # No blur needed

    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
         # Ensure shape is tuple
         shape_tuple = tuple(shape)
         return F.interpolate(x, size=shape_tuple, mode=interpolation_method, align_corners=align_corners)

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
          # Ensure shape is tuple
         shape_tuple = tuple(shape)
         return F.interpolate(x, size=shape_tuple, mode=interpolation_method, align_corners=align_corners)

    # --- Need update_optimizer_accum_grad_batches ---
    def update_optimizer_accum_grad_batches(self, batch_idx):
         # Requires access to self.trainer, usually set by PL automatically
         if not hasattr(self, 'trainer') or self.trainer is None:
              # Cannot determine accumulation logic without trainer state
              self.backprop_now = True # Default to backprop every step if trainer missing
              return

         # Determine total batches (handle different PL versions)
         try:
             # Newer PL versions might use different attributes
             if hasattr(self.trainer, 'num_training_batches'):
                  if isinstance(self.trainer.num_training_batches, int) and self.trainer.num_training_batches > 0:
                       self.total_batches_epoch = self.trainer.num_training_batches
                  elif hasattr(self.trainer, 'limit_train_batches') and isinstance(self.trainer.limit_train_batches, int):
                       self.total_batches_epoch = self.trainer.limit_train_batches
                  else: # Estimate from dataloader if possible
                       self.total_batches_epoch = len(self.trainer.train_dataloader)
             else: # Fallback for older versions
                  self.total_batches_epoch = len(self.trainer.train_dataloader)
         except:
              print("Warning: Could not determine total batches for gradient accumulation.")
              self.total_batches_epoch = batch_idx + 1 # Rough estimate

         if self.total_batches_epoch == float('inf'): # Handle infinite dataloaders
              self.total_batches_epoch = batch_idx + 100 # Arbitrary large number

         # Accumulation logic (simplified from original example)
         is_last_batch = (batch_idx + 1) == self.total_batches_epoch

         if self.current_optimizer_idx == 0: # Generator
              accum_steps = self.accumulate_grad_batches_g
              step_count = self.generator_step_count
         else: # Discriminator
              accum_steps = self.accumulate_grad_batches_d
              step_count = self.discriminator_step_count

         if is_last_batch: # Always step on last batch
              self.backprop_now = True
         elif (step_count + 1) % accum_steps == 0:
              self.backprop_now = True
         else:
              self.backprop_now = False


    # --- Need on_before_optimizer_step ---
    def on_before_optimizer_step(self, optimizer): # <<< REMOVE optimizer_idx
         # Apply gradient clipping
         # We clip ALL parameters associated with the *current* optimizer instance
         if self.grad_clip_val is not None and self.grad_clip_val > 0.0:
              # Get parameters associated with the passed optimizer
              params_to_clip = []
              for group in optimizer.param_groups:
                   params_to_clip.extend(group['params'])

              if params_to_clip:
                   # Directly clip the parameters linked to this optimizer
                   torch.nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip_val)
              else:
                   print(f"Warning: No parameters found for gradient clipping for the current optimizer.")

    # --- Need log_images3D ---
    @torch.no_grad()
    def log_images3D(self, batch, split="train", **kwargs):
         # Adapted for BiCond model
         log = dict()
         inputs = self.get_input(batch, self.image_key).to(self.device)
         target_voxel_size = self.get_input(batch, self.target_cond_key).to(self.device)

         log["target_voxel_size"] = f"{target_voxel_size[0].cpu().numpy()}" # Log first in batch

         if self.ae_mode == 'gt2gt':
             source_voxel_size = target_voxel_size
             lres_ims = inputs
             targets = inputs
             log["source_voxel_size"] = f"{source_voxel_size[0].cpu().numpy()}"

         elif self.ae_mode in ['deg2deg', 'deg2gt']:
             # Use fixed UFs for logging for consistency? Or random? Let's use random.
             uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False)
             ufs = [uf_h, uf_w, uf_z]
             source_voxel_size = self.calculate_source_voxel_size(target_voxel_size, ufs)
             lres_ims, targets = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
             log["source_voxel_size"] = f"{source_voxel_size[0].cpu().numpy()}"
             log["UFs"] = f"[{uf_h:.1f},{uf_w:.1f},{uf_z:.1f}]"
         else:
             raise NotImplementedError()

         # Generate reconstruction using both conditions
         xrec, _ = self(lres_ims,
                        source_cond_input=source_voxel_size,
                        target_cond_input=target_voxel_size)

         log["inputs"] = lres_ims       # Input to the network (potentially degraded)
         log["reconstructions"] = xrec  # Network output
         log["targets"] = targets       # Ground truth target for loss calculation

         return log


class AutoencoderKL3DV5(pl.LightningModule):
    """
    A 3D variant of AutoencoderKL that:
      - Processes inputs of shape [BS, 1, D, H, W]
      - Slices / downscales / upscales for deg2deg or deg2gt tasks
      - Uses an adversarial loss from the configured 'lossconfig'
      - Follows the same logging/naming scheme as the 2D version
      - Employs multiple optimizers in manual optimization
      
    Modified to include:
      - Manual frequency control for generator and discriminator updates.
      - Warm-up periods for the generator and discriminator.
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        trainconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        grad_clip_val=1.0,
        load_pretrained_weights_mode="both"
    ):
        super().__init__()
        self.automatic_optimization = False  
        self.current_mini_epoch = 0
        self.image_key = image_key
        self.load_pretrained_weights_mode = load_pretrained_weights_mode

        # Make sure Encoder3D_v3 and Decoder3D_v3 are available in the scope.
        self.encoder = Encoder3D_v3(**ddconfig)
        self.decoder = Decoder3D_v3(**ddconfig)

        # Instantiate the 3D loss/discriminator module.
        self.loss = instantiate_from_config(lossconfig)

        # Map from [2*z_channels, D', H', W'] -> [2*embed_dim, D', H', W'] and back.
        self.quant_conv = torch.nn.Conv3d(
            2 * ddconfig["z_channels"], 2 * embed_dim, kernel_size=1
        )
        self.post_quant_conv = torch.nn.Conv3d(
            embed_dim, ddconfig["z_channels"], kernel_size=1
        )

        # Additional config.
        self.downscaling_factors = trainconfig.get('downscaling_factors', [1])
        self.ae_mode = trainconfig.get('ae_mode', 'gt2gt')
        self.embed_dim = embed_dim
        
        # Grad accumulation and optimizer logic.
        self.current_optimizer_idx = 0
        self.generator_step_count = 0
        self.discriminator_step_count = 0

        self.accumulate_grad_batches_g = trainconfig.get('accumulate_grad_batches_g', 1)
        self.accumulate_grad_batches_d = trainconfig.get('accumulate_grad_batches_d', 1)
        self.lr_ratio_gtod = trainconfig.get('lr_ratio_gtod', 10)
        
        # --- Frequency Control ---
        self.generator_update_frequency = trainconfig.get('generator_update_frequency', 1)
        self.discriminator_update_frequency = trainconfig.get('discriminator_update_frequency', 1)
        self.g_updates_performed = 0
        self.d_updates_performed = 0

        # --- Warm-up Control ---
        self.generator_warmup_steps = trainconfig.get('generator_warmup_steps', 0)
        self.discriminator_warmup_steps = trainconfig.get('discriminator_warmup_steps', 0)
        
        print(f"--- Training Control Initialized ---")
        print(f"Generator: Update every {self.generator_update_frequency} cycle(s), Warm-up for {self.generator_warmup_steps} steps.")
        print(f"Discriminator: Update every {self.discriminator_update_frequency} cycle(s), Warm-up for {self.discriminator_warmup_steps} steps.")
        print(f"------------------------------------")

        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            if self.load_pretrained_weights_mode == "generator":
                ignore_keys = ignore_keys + ["loss.discriminator"]
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.grad_clip_val = grad_clip_val

    def on_save_checkpoint(self, checkpoint):
        checkpoint['current_mini_epoch'] = self.current_mini_epoch
        checkpoint['g_updates_performed'] = self.g_updates_performed
        checkpoint['d_updates_performed'] = self.d_updates_performed
        checkpoint['current_optimizer_idx'] = self.current_optimizer_idx

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading weights and training state from checkpoint...")
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)
        self.g_updates_performed = checkpoint.get('g_updates_performed', 0)
        self.d_updates_performed = checkpoint.get('d_updates_performed', 0)
        self.current_optimizer_idx = checkpoint.get('current_optimizer_idx', 0)
        print(f"Restored training state: g_updates_performed={self.g_updates_performed}, d_updates_performed={self.d_updates_performed}, current_optimizer_idx={self.current_optimizer_idx}")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict as per ignore_keys.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored 3D checkpoint from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution3D(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:  # [BS, D, H, W]
            x = x.unsqueeze(1)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()
    
    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def gaussian_kernel1d(self, sigma, size, device):
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        device = img.device
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)

        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, size_h, 1, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, size_w, 1)
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, 1, 1, size_z)

        img = F.conv3d(img, kernel_h, padding=(size_h // 2, 0, 0))
        img = F.conv3d(img, kernel_w, padding=(0, size_w // 2, 0))
        img = F.conv3d(img, kernel_z, padding=(0, 0, size_z // 2))
        return img

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0

        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)

        sigma_z *= (1 + aug_z)
        sigma_h *= (1 + aug_h)
        sigma_w *= (1 + aug_w)

        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
        target_vol = img
        bs, c, d, h, w = img.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        if is_train and degrade == 'blur':
            img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        shape = (d // uf_z, h // uf_h, w // uf_w)
        downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
        # Original logic returned the upscaled version for some modes
        upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return upscaled_vol, target_vol
    
    def _consistency_degrade(self, input, reconstruction, uf_h, uf_w, uf_z, is_train=True):
        
        bs, c, d, h, w = input.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        if is_train and degrade == 'blur':
            input_degraded = self.blur_tensor(input, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)
            reconstruction_degraded = self.blur_tensor(reconstruction, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        shape = (d // uf_z, h // uf_h, w // uf_w)
        input_degraded = self.downscale(input_degraded, shape, interpolation_method=interpolation_method, align_corners=True)
        reconstruction_degraded = self.downscale(reconstruction_degraded, shape, interpolation_method=interpolation_method, align_corners=True)
        # Original logic returned the upscaled version for some modes
        # upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return input_degraded, reconstruction_degraded

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        def weighted_choice(options):
            n = len(options)
            if n == 1:
                return options[0]
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)

        if is_train:
            if coupled_ufs:
                uf = weighted_choice(self.downscaling_factors)
                uf_h, uf_w, uf_z = uf, uf, uf
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else: # For validation/logging, use a fixed or representative factor
            if coupled_ufs:
                uf = self.downscaling_factors[len(self.downscaling_factors)//2]
                uf_h, uf_w, uf_z = uf, uf, uf
            else:
                # Use a specific, possibly different, factor for each dimension in validation
                uf_h = self.downscaling_factors[len(self.downscaling_factors)//2]
                uf_w = self.downscaling_factors[len(self.downscaling_factors)//2]
                uf_z = self.downscaling_factors[len(self.downscaling_factors)//2]
        return uf_h, uf_w, uf_z

    def get_vram_usage(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == 'cpu':
            return {"Alloc": 0, "Resv": 0}
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved  = torch.cuda.memory_reserved(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory
        alloc_percent = mem_allocated / total_mem * 100
        resv_percent  = mem_reserved / total_mem * 100
        log = {
            "Alloc": alloc_percent,
            "Resv":  resv_percent,
        }
        return log

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.generator_step_count = 0
            self.discriminator_step_count = 0
            if self.g_updates_performed < self.generator_update_frequency:
                self.current_optimizer_idx = 0
            else:
                self.current_optimizer_idx = 1

        opt_ae, opt_disc = self.optimizers()
        inputs = self.get_input(batch, self.image_key).to(torch.float)
        
        def an_optimizer_is_stepped(is_generator_step):
            """Manages the switching between optimizers based on update frequency."""
            if is_generator_step:
                self.g_updates_performed += 1
                if self.g_updates_performed >= self.generator_update_frequency:
                    self.current_optimizer_idx = 1
                    self.d_updates_performed = 0
                else:
                    self.current_optimizer_idx = 0
            else:
                self.d_updates_performed += 1
                if self.d_updates_performed >= self.discriminator_update_frequency:
                    self.current_optimizer_idx = 0
                    self.g_updates_performed = 0
                else:
                    self.current_optimizer_idx = 1
        
        can_train_g = self.global_step >= self.generator_warmup_steps
        can_train_d = self.global_step >= self.discriminator_warmup_steps
        
        # --- Logic for preparing inputs and targets based on ae_mode ---
        reconstructions, posterior, loss_target = None, None, None
        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)
            loss_target = inputs
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            loss_target = lres_ims
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            loss_target = hres_ims
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

        # --- Optimizer Steps ---
        if self.current_optimizer_idx == 0 and can_train_g:
            aeloss, log_dict_ae = self.loss(loss_target, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            
            self.manual_backward(aeloss / self.accumulate_grad_batches_g)
            self.generator_step_count += 1
            
            if self.generator_step_count >= self.accumulate_grad_batches_g:
                self.on_after_backward() # Call grad clipping
                opt_ae.step()
                opt_ae.zero_grad()
                self.generator_step_count = 0
                an_optimizer_is_stepped(is_generator_step=True)
            return aeloss

        elif self.current_optimizer_idx == 1 and can_train_d:
            discloss, log_dict_disc = self.loss(loss_target.detach(), reconstructions.detach(), posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(discloss / self.accumulate_grad_batches_d)
            self.discriminator_step_count += 1
            
            if self.discriminator_step_count >= self.accumulate_grad_batches_d:
                self.on_after_backward() # Call grad clipping
                opt_disc.step()
                opt_disc.zero_grad()
                self.discriminator_step_count = 0
                an_optimizer_is_stepped(is_generator_step=False)
            return discloss
        else:
            # Skip step if warming up
            if self.current_optimizer_idx == 0:
                an_optimizer_is_stepped(is_generator_step=True)
            elif self.current_optimizer_idx == 1:
                an_optimizer_is_stepped(is_generator_step=False)
            return None

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key).to(torch.float)
        
        # --- Logic for preparing inputs and targets based on ae_mode ---
        reconstructions, posterior, loss_target = None, None, None
        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)
            loss_target = inputs
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, posterior = self(lres_ims)
            loss_target = lres_ims
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, posterior = self(lres_ims)
            loss_target = hres_ims
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

        aeloss, log_dict_ae = self.loss(loss_target, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(loss_target, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="val")
        
        self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
        vram_log = self.get_vram_usage()
        self.log_dict({**vram_log, **log_dict_ae, **log_dict_disc})
        return self.log_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        lr = self.learning_rate
        
        params_ae = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                    list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        opt_ae = torch.optim.AdamW(params_ae, lr=lr, betas=(0.5, 0.9))

        params_disc = self.loss.discriminator.parameters()
        opt_disc = torch.optim.AdamW(params_disc, lr=lr / self.lr_ratio_gtod, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_after_backward(self):
        if self.grad_clip_val is not None and self.grad_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

    @torch.no_grad()
    def log_images3D(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device).to(torch.float)

        if self.ae_mode == 'gt2gt':
            xrec, _ = self(x)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=False)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=False)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims
            
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
    
class AutoencoderKL3DV6(pl.LightningModule):
    """
    A 3D variant of AutoencoderKL that:
      - Processes inputs of shape [BS, 1, D, H, W]
      - Slices / downscales / upscales for deg2deg or deg2gt tasks
      - Uses an adversarial loss from the configured 'lossconfig'
      - Follows the same logging/naming scheme as the 2D version
      - Employs multiple optimizers in manual optimization
      
    Modified to include:
      - Manual frequency control for generator and discriminator updates.
      - Warm-up periods for the generator and discriminator.
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        lossconfig_consistency,
        trainconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        remap=None,
        sane_index_shape=False,
        grad_clip_val=1.0,
        load_pretrained_weights_mode="both",
        use_consistency_degrade=True
    ):
        super().__init__()
        self.automatic_optimization = False  
        self.current_mini_epoch = 0
        self.image_key = image_key
        self.load_pretrained_weights_mode = load_pretrained_weights_mode
        self.use_consistency_degrade = use_consistency_degrade

        # Make sure Encoder3D_v3 and Decoder3D_v3 are available in the scope.
        self.encoder = Encoder3D_v3(**ddconfig)
        self.decoder = Decoder3D_v3(**ddconfig)

        # Instantiate the 3D loss/discriminator module.
        self.loss = instantiate_from_config(lossconfig)
        self.loss_consistency = instantiate_from_config(lossconfig_consistency)

        # Map from [2*z_channels, D', H', W'] -> [2*embed_dim, D', H', W'] and back.
        self.quant_conv = torch.nn.Conv3d(
            2 * ddconfig["z_channels"], 2 * embed_dim, kernel_size=1
        )
        self.post_quant_conv = torch.nn.Conv3d(
            embed_dim, ddconfig["z_channels"], kernel_size=1
        )

        # Additional config.
        self.downscaling_factors = trainconfig.get('downscaling_factors', [1])
        self.ae_mode = trainconfig.get('ae_mode', 'gt2gt')
        self.embed_dim = embed_dim
        
        # Grad accumulation and optimizer logic.
        self.current_optimizer_idx = 0
        self.generator_step_count = 0
        self.discriminator_step_count = 0

        self.accumulate_grad_batches_g = trainconfig.get('accumulate_grad_batches_g', 1)
        self.accumulate_grad_batches_d = trainconfig.get('accumulate_grad_batches_d', 1)
        self.lr_ratio_gtod = trainconfig.get('lr_ratio_gtod', 10)
        
        # --- Frequency Control ---
        self.generator_update_frequency = trainconfig.get('generator_update_frequency', 1)
        self.discriminator_update_frequency = trainconfig.get('discriminator_update_frequency', 1)
        self.g_updates_performed = 0
        self.d_updates_performed = 0

        # --- Warm-up Control ---
        self.generator_warmup_steps = trainconfig.get('generator_warmup_steps', 0)
        self.discriminator_warmup_steps = trainconfig.get('discriminator_warmup_steps', 0)
        
        print(f"--- Training Control Initialized ---")
        print(f"Generator: Update every {self.generator_update_frequency} cycle(s), Warm-up for {self.generator_warmup_steps} steps.")
        print(f"Discriminator: Update every {self.discriminator_update_frequency} cycle(s), Warm-up for {self.discriminator_warmup_steps} steps.")
        print(f"------------------------------------")

        if colorize_nlabels is not None:
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))

        if monitor is not None:
            self.monitor = monitor

        if ckpt_path is not None:
            if self.load_pretrained_weights_mode == "generator":
                ignore_keys = ignore_keys + ["loss.discriminator"]
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.grad_clip_val = grad_clip_val

    def on_save_checkpoint(self, checkpoint):
        checkpoint['current_mini_epoch'] = self.current_mini_epoch
        checkpoint['g_updates_performed'] = self.g_updates_performed
        checkpoint['d_updates_performed'] = self.d_updates_performed
        checkpoint['current_optimizer_idx'] = self.current_optimizer_idx

    def on_load_checkpoint(self, checkpoint):
        print(f"Loading weights and training state from checkpoint...")
        self.current_mini_epoch = checkpoint.get('current_mini_epoch', 0)
        self.g_updates_performed = checkpoint.get('g_updates_performed', 0)
        self.d_updates_performed = checkpoint.get('d_updates_performed', 0)
        self.current_optimizer_idx = checkpoint.get('current_optimizer_idx', 0)
        print(f"Restored training state: g_updates_performed={self.g_updates_performed}, d_updates_performed={self.d_updates_performed}, current_optimizer_idx={self.current_optimizer_idx}")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in list(sd.keys()):
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict as per ignore_keys.")
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored 3D checkpoint from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution3D(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 4:  # [BS, D, H, W]
            x = x.unsqueeze(1)
        x = x.to(memory_format=torch.contiguous_format)
        return x.float()
    
    def downscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def upscale(self, x, shape, interpolation_method='trilinear', align_corners=True):
        return F.interpolate(
            x, size=shape, mode=interpolation_method, align_corners=align_corners
        )

    def gaussian_kernel1d(self, sigma, size, device):
        x = torch.arange(-size // 2 + 1, size // 2 + 1, device=device)
        g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def gaussian_blur(self, img, sigma_z, sigma_h, sigma_w):
        device = img.device
        size_z = int(2 * math.ceil(2 * sigma_z) + 1)
        size_h = int(2 * math.ceil(2 * sigma_h) + 1)
        size_w = int(2 * math.ceil(2 * sigma_w) + 1)

        kernel_h = self.gaussian_kernel1d(sigma_h, size_h, device).view(1, 1, size_h, 1, 1)
        kernel_w = self.gaussian_kernel1d(sigma_w, size_w, device).view(1, 1, 1, size_w, 1)
        kernel_z = self.gaussian_kernel1d(sigma_z, size_z, device).view(1, 1, 1, 1, size_z)

        img = F.conv3d(img, kernel_h, padding=(size_h // 2, 0, 0))
        img = F.conv3d(img, kernel_w, padding=(0, size_w // 2, 0))
        img = F.conv3d(img, kernel_z, padding=(0, 0, size_z // 2))
        return img

    def blur_tensor(self, img, uf_h=1.0, uf_w=1.0, uf_z=1.0, p_aug_sigma=0.5):
        sigma_z = uf_z / 2.0
        sigma_h = uf_h / 2.0
        sigma_w = uf_w / 2.0

        aug_z = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_h = random.uniform(-p_aug_sigma, p_aug_sigma)
        aug_w = random.uniform(-p_aug_sigma, p_aug_sigma)

        sigma_z *= (1 + aug_z)
        sigma_h *= (1 + aug_h)
        sigma_w *= (1 + aug_w)

        img = self.gaussian_blur(img, sigma_z, sigma_h, sigma_w)
        return img

    def prepare_input_target_torch(self, img, uf_h, uf_w, uf_z, is_train=True):
        target_vol = img
        bs, c, d, h, w = img.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        if is_train and degrade == 'blur':
            img = self.blur_tensor(img, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)

        shape = (d // uf_z, h // uf_h, w // uf_w)
        downscaled_vol = self.downscale(img, shape, interpolation_method=interpolation_method, align_corners=True)
        # Original logic returned the upscaled version for some modes
        upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return upscaled_vol, target_vol
    
    def _consistency_degrade(self, input, reconstruction, uf_h, uf_w, uf_z, is_train=True):
        
        bs, c, d, h, w = input.shape
        interpolation_method = 'trilinear'
        degrade = random.choice(['blur', 'noblur'])

        if is_train and degrade == 'blur':
            input_degraded = self.blur_tensor(input, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)
            reconstruction_degraded = self.blur_tensor(reconstruction, uf_z=uf_z, uf_h=uf_h, uf_w=uf_w)
        else:
            input_degraded = input
            reconstruction_degraded = reconstruction

        shape = (d // uf_z, h // uf_h, w // uf_w)
        input_degraded = self.downscale(input_degraded, shape, interpolation_method=interpolation_method, align_corners=True)
        reconstruction_degraded = self.downscale(reconstruction_degraded, shape, interpolation_method=interpolation_method, align_corners=True)
        # Original logic returned the upscaled version for some modes
        # upscaled_vol = self.upscale(downscaled_vol, (d, h, w), interpolation_method='trilinear')
        return input_degraded, reconstruction_degraded

    def get_batch_ufs_and_noise_std(self, is_train=True, coupled_ufs=False):
        def weighted_choice(options):
            n = len(options)
            if n == 1:
                return options[0]
            probabilities = [0.1] + [0.9 / (n - 1)] * (n - 1)
            return np.random.choice(options, p=probabilities)

        if is_train:
            if coupled_ufs:
                uf = weighted_choice(self.downscaling_factors)
                uf_h, uf_w, uf_z = uf, uf, uf
            else:
                uf_h = weighted_choice(self.downscaling_factors)
                uf_w = weighted_choice(self.downscaling_factors)
                uf_z = weighted_choice(self.downscaling_factors)
        else: # For validation/logging, use a fixed or representative factor
            if coupled_ufs:
                uf = self.downscaling_factors[len(self.downscaling_factors)//2]
                uf_h, uf_w, uf_z = uf, uf, uf
            else:
                # Use a specific, possibly different, factor for each dimension in validation
                uf_h = self.downscaling_factors[len(self.downscaling_factors)//2]
                uf_w = self.downscaling_factors[len(self.downscaling_factors)//2]
                uf_z = self.downscaling_factors[len(self.downscaling_factors)//2]
        return uf_h, uf_w, uf_z

    def get_vram_usage(self):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if device.type == 'cpu':
            return {"Alloc": 0, "Resv": 0}
        mem_allocated = torch.cuda.memory_allocated(device)
        mem_reserved  = torch.cuda.memory_reserved(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory
        alloc_percent = mem_allocated / total_mem * 100
        resv_percent  = mem_reserved / total_mem * 100
        log = {
            "Alloc": alloc_percent,
            "Resv":  resv_percent,
        }
        return log

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.generator_step_count = 0
            self.discriminator_step_count = 0
            if self.g_updates_performed < self.generator_update_frequency:
                self.current_optimizer_idx = 0
            else:
                self.current_optimizer_idx = 1

        opt_ae, opt_disc = self.optimizers()
        inputs = self.get_input(batch, self.image_key).to(torch.float)
        
        def an_optimizer_is_stepped(is_generator_step):
            """Manages the switching between optimizers based on update frequency."""
            if is_generator_step:
                self.g_updates_performed += 1
                if self.g_updates_performed >= self.generator_update_frequency:
                    self.current_optimizer_idx = 1
                    self.d_updates_performed = 0
                else:
                    self.current_optimizer_idx = 0
            else:
                self.d_updates_performed += 1
                if self.d_updates_performed >= self.discriminator_update_frequency:
                    self.current_optimizer_idx = 0
                    self.g_updates_performed = 0
                else:
                    self.current_optimizer_idx = 1
        
        can_train_g = self.global_step >= self.generator_warmup_steps
        can_train_d = self.global_step >= self.discriminator_warmup_steps
        
        # --- Logic for preparing inputs and targets based on ae_mode ---
        reconstructions, posterior, loss_target = None, None, None
        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)
            loss_target = inputs
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            loss_target = lres_ims
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=True, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=True)
            reconstructions, posterior = self(lres_ims)
            if self.use_consistency_degrade:
                true_lres_ims, fake_lres_ims = self._consistency_degrade(inputs, reconstructions, uf_h, uf_w, uf_z, is_train=True)
            loss_target = hres_ims
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

        # --- Optimizer Steps ---
        if self.current_optimizer_idx == 0 and can_train_g:
            aeloss, log_dict_ae = self.loss(loss_target, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="train")
            aeloss_consistency, log_dict_ae_consistency = self.loss_consistency(true_lres_ims, fake_lres_ims, 0, self.global_step, split="train") if self.use_consistency_degrade else (0.0, {})
            aeloss += aeloss_consistency
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log, **log_dict_ae}, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("train/aeloss_consistency", aeloss_consistency, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.manual_backward(aeloss / self.accumulate_grad_batches_g)
            self.generator_step_count += 1
            
            if self.generator_step_count >= self.accumulate_grad_batches_g:
                self.on_after_backward() # Call grad clipping
                opt_ae.step()
                opt_ae.zero_grad()
                self.generator_step_count = 0
                an_optimizer_is_stepped(is_generator_step=True)
            return aeloss

        elif self.current_optimizer_idx == 1 and can_train_d:
            discloss, log_dict_disc = self.loss(loss_target.detach(), reconstructions.detach(), posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="train")
            vram_log = self.get_vram_usage()
            self.log_dict({**vram_log, **log_dict_disc}, prog_bar=True, on_step=True, on_epoch=True)
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

            self.manual_backward(discloss / self.accumulate_grad_batches_d)
            self.discriminator_step_count += 1
            
            if self.discriminator_step_count >= self.accumulate_grad_batches_d:
                self.on_after_backward() # Call grad clipping
                opt_disc.step()
                opt_disc.zero_grad()
                self.discriminator_step_count = 0
                an_optimizer_is_stepped(is_generator_step=False)
            return discloss
        else:
            # Skip step if warming up
            if self.current_optimizer_idx == 0:
                an_optimizer_is_stepped(is_generator_step=True)
            elif self.current_optimizer_idx == 1:
                an_optimizer_is_stepped(is_generator_step=False)
            return None

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key).to(torch.float)
        
        # --- Logic for preparing inputs and targets based on ae_mode ---
        reconstructions, posterior, loss_target = None, None, None
        if self.ae_mode == 'gt2gt':
            reconstructions, posterior = self(inputs)
            loss_target = inputs
        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, posterior = self(lres_ims)
            loss_target = lres_ims
        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(inputs, uf_h, uf_w, uf_z, is_train=False)
            reconstructions, posterior = self(lres_ims)
            loss_target = hres_ims
        else:
            raise NotImplementedError(f"Unknown ae_mode: {self.ae_mode}")

        aeloss, log_dict_ae = self.loss(loss_target, reconstructions, posterior, 0, self.global_step, last_layer=self.get_last_layer(), split="val")
        discloss, log_dict_disc = self.loss(loss_target, reconstructions, posterior, 1, self.global_step, last_layer=self.get_last_layer(), split="val")
        
        self.log("val_rec_loss", log_dict_ae.get("val/nll_loss", aeloss))
        vram_log = self.get_vram_usage()
        self.log_dict({**vram_log, **log_dict_ae, **log_dict_disc})
        return self.log_dict

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        lr = self.learning_rate
        
        params_ae = list(self.encoder.parameters()) + list(self.decoder.parameters()) + \
                    list(self.quant_conv.parameters()) + list(self.post_quant_conv.parameters())
        opt_ae = torch.optim.AdamW(params_ae, lr=lr, betas=(0.5, 0.9))

        params_disc = self.loss.discriminator.parameters()
        opt_disc = torch.optim.AdamW(params_disc, lr=lr / self.lr_ratio_gtod, betas=(0.5, 0.9))
        
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def on_after_backward(self):
        if self.grad_clip_val is not None and self.grad_clip_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_val)

    @torch.no_grad()
    def log_images3D(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key).to(self.device).to(torch.float)

        if self.ae_mode == 'gt2gt':
            xrec, _ = self(x)
            log["inputs"] = x
            log["reconstructions"] = xrec
            log["targets"] = x

        elif self.ae_mode == 'deg2deg':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, _ = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=False)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = lres_ims

        elif self.ae_mode == 'deg2gt':
            uf_h, uf_w, uf_z = self.get_batch_ufs_and_noise_std(is_train=False, coupled_ufs=False)
            lres_ims, hres_ims = self.prepare_input_target_torch(x, uf_h, uf_w, uf_z, is_train=False)
            xrec, _ = self(lres_ims)
            log["inputs"] = lres_ims
            log["reconstructions"] = xrec
            log["targets"] = hres_ims
            
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
        return x
    
