import torch
import torch.nn as nn
import torch.nn.functional as F

from taming.modules.losses.lpips import LPIPS
from taming.modules.discriminator.model import NLayerDiscriminator, NLayerDiscriminator3D, weights_init


class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm,
                                                 ndf=disc_ndf
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])

        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss)

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()
            #loss = 1.3760 + 0.0 * 0.0 * -0.0103 + 1.0 * 0.4483
            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log

class VQLPIPSWithDiscriminator3D(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_slices = 32):
        """
        A 3D-adapted VQ-GAN loss with LPIPS on each 2D slice of a 3D volume [BS, C, H, W, Z].
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.n_slices = n_slices
        print(f'Number of slices for loss function: {n_slices}')

        # LPIPS expects 2D images [BS, C, H, W]. We'll handle 3D by slicing in forward().
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

        print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute the weight for the discriminator feature matching term.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
    
    def extract_subset_slices_from3d(self, in_slices, rec_slices, n_slices = 16):
        # Get a random permutation of indices from 0 to 287, then take the first n_slices indices
        indices = torch.randperm(in_slices.shape[0])[:n_slices]

        # Use these indices to select slices from the tensor along the first dimension
        in_slices = in_slices[indices]
        rec_slices = rec_slices[indices]

        return in_slices, rec_slices

    def to_slices_subset(self, inputs, reconstructions, n_slices = 16):
        """
        Slices the input and reconstruction volumes into XY, XZ, YZ planes. The final shape will be [n_slices*3, C, D, D]
        """
        BS, C, D, D2, D3 = inputs.shape
        assert D == D2 == D3, "Expected a cubic volume [BS, C, D, D, D]."

        # ---------- XY slices (fix z in [0..D-1]) -----------
        # rearrange => [BS, D, C, D, D], then flatten => [BS*D, C, D, D]
        xy_slices_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D, C, D, D)
        xy_slices_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D, C, D, D)
        # extract subsample of n_slices
        xy_slices_in, xy_slices_rec = self.extract_subset_slices_from3d(xy_slices_in, xy_slices_rec, n_slices = n_slices)

        # ---------- XZ slices (fix w in [0..D-1]) -----------
        # rearrange => [BS, D, C, D, D], then flatten => [BS*D, C, D, D]
        xz_slices_in = inputs.permute(0, 3, 1, 2, 4).reshape(BS * D, C, D, D)
        xz_slices_rec = reconstructions.permute(0, 3, 1, 2, 4).reshape(BS * D, C, D, D)
        # extract subsample of n_slices
        xz_slices_in, xz_slices_rec = self.extract_subset_slices_from3d(xz_slices_in, xz_slices_rec, n_slices = n_slices)

        # ---------- YZ slices (fix h in [0..D-1]) -----------
        # rearrange => [BS, D, C, D, D], then flatten => [BS*D, C, D, D]
        yz_slices_in = inputs.permute(0, 2, 1, 3, 4).reshape(BS * D, C, D, D)
        yz_slices_rec = reconstructions.permute(0, 2, 1, 3, 4).reshape(BS * D, C, D, D)
        # extract subsample of n_slices
        yz_slices_in, yz_slices_rec = self.extract_subset_slices_from3d(yz_slices_in, yz_slices_rec, n_slices = n_slices)

        # Concatenate them along the batch dimension => [BS*3*D, C, D, D]
        in_slices = torch.cat([xy_slices_in, xz_slices_in, yz_slices_in], dim=0)
        rec_slices = torch.cat([xy_slices_rec, xz_slices_rec, yz_slices_rec], dim=0)

        return in_slices, rec_slices

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        """
        inputs, reconstructions: [BS, C, H, W, Z]
          - We loop over all slices along 3 planes (XY, XZ, YZ) to compute LPIPS.
        codebook_loss: Codebook commitment loss (VQ-GAN).
        optimizer_idx: 0 -> generator update, 1 -> discriminator update.
        """
        # 1) Pixel-level reconstruction loss (L1)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # 2) Perceptual loss via LPIPS, looping over 3D slices
        if self.perceptual_weight > 0:
            #Extracts n_slices slices from the input and reconstruction volumes randomly
            in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions, n_slices = self.n_slices)

            # Single LPIPS call
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices).mean() #0.7046

            # Add to pixel-wise rec_loss (broadcast over 5D)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # 3) Combine into total reconstruction loss
        nll_loss = torch.mean(rec_loss)  # scalar #1.564

        # --------------------- GAN Part ---------------------
        if optimizer_idx == 0:
            # Generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(rec_slices.contiguous())
            else:
                assert self.disc_conditional
                # condition: concat along channel dim
                logits_fake = self.discriminator(torch.cat((rec_slices.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake) #-0.6709

            # Adaptive weight for the discriminator's feature matching
            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                # e.g. if there's a grad issue or in eval mode
                d_weight = torch.tensor(0.0, device=inputs.device) #0.0

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # Final generator loss: 1.567 + 0.0 * 0.0 * -0.6709 + 1.0 * 0.1228
            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/quant_loss": codebook_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/p_loss": p_loss_3d.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/g_loss": g_loss.detach().mean(),
            }
            return loss, log

        elif optimizer_idx == 1:
            # Discriminator update
            if cond is None:
                logits_real = self.discriminator(in_slices.contiguous().detach())
                logits_fake = self.discriminator(rec_slices.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((in_slices.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((rec_slices.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log