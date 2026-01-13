import torch
import torch.nn as nn

from taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?


class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous(), spatial_avg=True)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss #Shape is [8, 1, 192, 192], mean 2.2146
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0] #value of 81639.0859
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]#value of 81639.0859
        kl_loss = posteriors.kl() #shape is [8], mean of 124.13
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous()) #Shape torch.Size([8, 1, 22, 22]), mean tensor(-0.0140, device='cuda:0')
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake) #Value of 0.0140

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
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


class LPIPSWithDiscriminator3D(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", n_slices = 32):
        """
        A 3D loss module that:
          - Computes pixel (L1) recon loss
          - Slices each volume into XY, XZ, YZ planes
          - Stacks those slices into a single batch for one LPIPS call
          - Optionally includes a 3D discriminator for adversarial training

        Expects volumes of shape [BS, C, D, D, D].
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"
        print("---------This is the right discriminator")
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.n_slices = n_slices
        print(f'Number of slices for loss function: {n_slices}')

        # LPIPS is originally 2D; we'll flatten slices for a single call
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # Learnable log-variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # 3D discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the discriminator feature matching term.
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
        
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        """
        Args:
          inputs:          [BS, C, D, D, D]
          reconstructions: [BS, C, D, D, D]
          posteriors:      Some distribution with .kl() method
          optimizer_idx:   0 => generator update, 1 => discriminator update
          global_step:     current training step
          last_layer:      optional last layer for adaptive weight
          cond:            optional conditioning volume if disc_conditional=True
          split:           'train' or 'val'
          weights:         optional voxel-wise weighting
        """
        # 1) Pixel-level L1 loss
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        # 2) Perceptual loss: slice along XY, XZ, YZ planes and do ONE LPIPS call
        #    for shape [BS * 3 * D, C, D, D]
        if self.perceptual_weight > 0:
            #Extracts n_slices slices from the input and reconstruction volumes randomly
            in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions, n_slices = self.n_slices)

            # Single LPIPS call
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=True).mean()

            # Add to pixel-wise rec_loss (broadcast over 5D)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # 3) Negative log-likelihood with log-var
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        if weights is not None:
            nll_loss = nll_loss * weights
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0] #Modify since nll_loss is too big in 3D tensors
        nll_loss = torch.sum(nll_loss) / (nll_loss.shape[0]*nll_loss.shape[-1]) #This new rescaling is to take into account the increase in size

        # For logging: store unweighted reconstruction loss
        # unweighted_nll = torch.sum(rec_loss) / rec_loss.shape[0]
        unweighted_nll = torch.sum(rec_loss) / (rec_loss.shape[0]*rec_loss.shape[-1])

        # 4) KL loss
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = torch.sum(kl_loss) / (kl_loss.shape[0]*kl_loss.shape[-1])

        # -----------------  GAN PART  -----------------
        if optimizer_idx == 0:
            # Generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(rec_slices.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((rec_slices.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)
            g_loss = g_loss.to(torch.float)
            # Possibly compute adaptive weight for the generator
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": unweighted_nll.detach().mean(),
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
                logits_real = self.discriminator(
                    torch.cat((in_slices.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((rec_slices.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log

class LPIPSWithDiscriminator3D_v2(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", n_slices = 32):
        """
        A 3D loss module that:
          - Computes pixel (L1) recon loss
          - Slices each volume into XY, XZ, YZ planes
          - Stacks those slices into a single batch for one LPIPS call
          - Optionally includes a 3D discriminator for adversarial training

        Expects volumes of shape [BS, C, D, D, D].
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"
        print("---------This is the right discriminator")
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.n_slices = n_slices
        print(f'Number of slices for loss function: {n_slices}')

        # LPIPS is originally 2D; we'll flatten slices for a single call
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # Learnable log-variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # 3D discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the discriminator feature matching term.
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

    def to_slices_subset(self, inputs, reconstructions, n_slices = 64):
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
        
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        """
        Args:
          inputs:          [BS, C, D, D, D]
          reconstructions: [BS, C, D, D, D]
          posteriors:      Some distribution with .kl() method
          optimizer_idx:   0 => generator update, 1 => discriminator update
          global_step:     current training step
          last_layer:      optional last layer for adaptive weight
          cond:            optional conditioning volume if disc_conditional=True
          split:           'train' or 'val'
          weights:         optional voxel-wise weighting
        """
        # 1) Pixel-level L1 loss
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions, n_slices = self.n_slices) #Shape [96, 1, 96, 96]
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous()) # torch.Size([192, 1, 1, 96, 96])

        # 2) Perceptual loss: slice along XY, XZ, YZ planes and do ONE LPIPS call
        #    for shape [BS * 3 * D, C, D, D]
        if self.perceptual_weight > 0:
            # Single LPIPS call
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg = False) # torch.Size([192, 1, 1, 1])

            # Add to pixel-wise rec_loss (broadcast over 5D)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # 3) Negative log-likelihood with log-var
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        if weights is not None:
            nll_loss = nll_loss * weights
        # nll_loss = torch.sum(nll_loss) / nll_loss.shape[0] #Modify since nll_loss is too big in 3D tensors
        nll_loss = torch.sum(nll_loss) / (nll_loss.shape[0]) #This new rescaling is to take into account the increase in size

        # For logging: store unweighted reconstruction loss
        # unweighted_nll = torch.sum(rec_loss) / rec_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / (rec_loss.shape[0])

        # 4) KL loss
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = torch.sum(kl_loss) / (kl_loss.shape[0])

        # -----------------  GAN PART  -----------------
        if optimizer_idx == 0:
            # Generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(rec_slices.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((rec_slices.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)
            g_loss = g_loss.to(torch.float)
            # Possibly compute adaptive weight for the generator
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
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
                logits_real = self.discriminator(
                    torch.cat((in_slices.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((rec_slices.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log
        
class LPIPSWithDiscriminator3D_v3(nn.Module):
    def __init__(self, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge", n_slices = 96):
        """
        A 3D loss module that:
          - Computes pixel (L1) recon loss
          - Slices each volume into XY, XZ, YZ planes
          - Stacks those slices into a single batch for one LPIPS call
          - Optionally includes a 3D discriminator for adversarial training

        Expects volumes of shape [BS, C, D, D, D].
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"
        print("---------This is the right discriminator")
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.n_slices = n_slices
        print(f'Number of slices for loss function: {n_slices}')

        # LPIPS is originally 2D; we'll flatten slices for a single call
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # Learnable log-variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # 3D discriminator
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the discriminator feature matching term.
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

    def to_slices_subset(self, inputs, reconstructions, n_slices = 64):
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
        
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        """
        Args:
          inputs:          [BS, C, D, D, D]
          reconstructions: [BS, C, D, D, D]
          posteriors:      Some distribution with .kl() method
          optimizer_idx:   0 => generator update, 1 => discriminator update
          global_step:     current training step
          last_layer:      optional last layer for adaptive weight
          cond:            optional conditioning volume if disc_conditional=True
          split:           'train' or 'val'
          weights:         optional voxel-wise weighting
        """
        # 1) Pixel-level L1 loss
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions, n_slices = self.n_slices) #Shape [96, 1, 96, 96]
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous()) # torch.Size([192, 1, 1, 96, 96])

        # 2) Perceptual loss: slice along XY, XZ, YZ planes and do ONE LPIPS call
        #    for shape [BS * 3 * D, C, D, D]
        if self.perceptual_weight > 0:
            # Single LPIPS call
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg = False) # torch.Size([192, 1, 1, 1])

            # Add to pixel-wise rec_loss (broadcast over 5D)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # 3) Negative log-likelihood with log-var
        nll_loss = rec_loss
        #nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        nll_loss = torch.mean(nll_loss) #nll_loss = 1.5770

        # 4) KL loss
        kl_loss = posteriors.kl()
        # kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        kl_loss = torch.sum(kl_loss) / (kl_loss.shape[0])

        # -----------------  GAN PART  -----------------
        if optimizer_idx == 0:
            # Generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(rec_slices.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(
                    torch.cat((rec_slices.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)
            g_loss = g_loss.to(torch.float)
            # Possibly compute adaptive weight for the generator
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/logvar": self.logvar.detach(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
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
                logits_real = self.discriminator(
                    torch.cat((in_slices.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((rec_slices.contiguous().detach(), cond), dim=1)
                )

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log
        
class LPIPSWithDiscriminator3D_v4(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 n_slices=32):
        """
        A 3D loss module that:
          - Slices each [BS, C, D, D, D] volume into 2D planes (XY, XZ, YZ)
          - Computes L1 + LPIPS on those slices (as in the 2D version)
          - Includes a 2D NLayerDiscriminator for adversarial training on slices
          - Uses the same logvar weighting & kl_weight logic as LPIPSWithDiscriminator

        Args:
          disc_start (int): iteration at which to start adversarial training
          logvar_init (float): initial log-variance for NLL weighting
          kl_weight (float): how much to scale the KL term
          pixelloss_weight (float): optional multiplier for pixel (L1) term
          disc_num_layers (int): how many layers in the NLayerDiscriminator
          disc_in_channels (int): input channels for discriminator (3 if rgb, 1 if grayscale)
          disc_factor (float): overall multiplier for adversarial loss
          disc_weight (float): factor for the adaptive weight
          perceptual_weight (float): weight for the LPIPS term
          use_actnorm (bool): use ActNorm in the discriminator instead of BatchNorm
          disc_conditional (bool): if True, pass `cond` into discriminator as extra channels
          disc_loss (str): "hinge" or "vanilla" (BCE) for the adversarial criterion
          n_slices (int): how many slices to sample per orientation (XY, XZ, YZ)
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # -- Main weights and hyperparams --
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        # -- Slicing config --
        self.n_slices = n_slices
        print(f"Number of slices for 3D volume: {n_slices}")

        # -- LPIPS (2D) --
        #   Use .eval() so it doesn’t update internal stats
        self.perceptual_loss = LPIPS().eval()

        # -- Learnable log-variance for nll_loss
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # -- 2D Discriminator & related config --
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term
        based on ratio of reconstruction grads to generator grads.
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

    def extract_subset_slices_from3d(self, in_slices, rec_slices, n_slices=16):
        """
        Randomly pick n_slices from the total slices along a given orientation.
        """
        total_slices = in_slices.shape[0]
        indices = torch.randperm(total_slices, device=in_slices.device)[:n_slices]

        in_slices = in_slices[indices]
        rec_slices = rec_slices[indices]
        return in_slices, rec_slices

    def to_slices_subset(self, inputs, reconstructions, n_slices=64):
        """
        Convert 3D volumes [BS, C, D, D, D] into a subset of 2D slices for XY, XZ, YZ.

        Returns:
          in_slices:  [N, C, H, W]
          rec_slices: [N, C, H, W]
        where N = n_slices * 3 for each batch entry (roughly).
        """
        BS, C, D, D2, D3 = inputs.shape
        assert (D == D2 == D3), "Expected cubic volume [BS, C, D, D, D]."

        # 1) XY slices (z in [0..D-1])
        #    shape => [BS*D, C, D, D]
        xy_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D, C, D, D)
        xy_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D, C, D, D)
        xy_in, xy_rec = self.extract_subset_slices_from3d(xy_in, xy_rec, n_slices=n_slices)

        # 2) XZ slices (w in [0..D-1])
        #    shape => [BS*D, C, D, D]
        xz_in = inputs.permute(0, 3, 1, 2, 4).reshape(BS * D, C, D, D)
        xz_rec = reconstructions.permute(0, 3, 1, 2, 4).reshape(BS * D, C, D, D)
        xz_in, xz_rec = self.extract_subset_slices_from3d(xz_in, xz_rec, n_slices=n_slices)

        # 3) YZ slices (h in [0..D-1])
        #    shape => [BS*D, C, D, D]
        yz_in = inputs.permute(0, 2, 1, 3, 4).reshape(BS * D, C, D, D)
        yz_rec = reconstructions.permute(0, 2, 1, 3, 4).reshape(BS * D, C, D, D)
        yz_in, yz_rec = self.extract_subset_slices_from3d(yz_in, yz_rec, n_slices=n_slices)

        # Concatenate along batch dim => [N, C, H, W]
        in_slices = torch.cat([xy_in, xz_in, yz_in], dim=0)
        rec_slices = torch.cat([xy_rec, xz_rec, yz_rec], dim=0)

        return in_slices, rec_slices

    def forward(self,
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train",
                weights=None):
        """
        Args:
          inputs:          [BS, C, D, D, D]
          reconstructions: [BS, C, D, D, D]
          posteriors:      object with .kl() method => KL of latent distribution
          optimizer_idx:   0 => generator step, 1 => discriminator step
          global_step:     current training step (for disc_factor scheduling)
          last_layer:      optional last layer for adaptive weight
          cond:            optional 2D conditioning appended as extra channels
          split:           'train' or 'val'
          weights:         optional weighting map

        Returns:
          (loss, log_dict) for whichever optimizer_idx is active.
        """

        # -------------------------------------------------
        # 1) Extract 2D slices for L1 + LPIPS
        # -------------------------------------------------
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions,
                                                      n_slices=self.n_slices)
        # rec_slices shape => [N, C, H, W], typically N = 3 * BS * n_slices

        # Pixel-level L1, then reduce over (C,H,W)
        rec_loss = torch.abs(in_slices - rec_slices).mean(dim=[1,2,3], keepdim=True)
        # shape => [N,1,1,1]

        # If using LPIPS:
        if self.perceptual_weight > 0:
            # same approach as the 2D code: set spatial_avg=True
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=True)
            # p_loss_3d => [N,1,1,1]
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # -------------------------------------------------
        # 2) Combine into negative log-likelihood with logvar
        #    nll_loss = rec_loss / exp(logvar) + logvar
        # -------------------------------------------------
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        # shape => [N,1,1,1]

        # Optional voxel/ slice weighting
        if weights is not None:
            # ensure weights has shape [N,1,1,1] or broadcasts
            nll_loss = nll_loss * weights

        # For logging, store the unweighted average, as in 2D:
        unweighted_nll_loss = nll_loss.mean()  # overall mean

        # We'll do the final "weighted_nll_loss" as well:
        weighted_nll_loss = nll_loss.sum() / nll_loss.shape[0]  # same as .mean()

        # -------------------------------------------------
        # 3) KL Loss
        # -------------------------------------------------
        kl = posteriors.kl()
        kl_loss = torch.sum(kl) / kl.shape[0]

        # -------------------------------------------------
        # 4) If Generator Update (optimizer_idx=0)
        # -------------------------------------------------
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional, "cond=None but disc_conditional=True?"
                logits_fake = self.discriminator(rec_slices)
            else:
                assert self.disc_conditional, "cond is provided but disc_conditional=False?"
                logits_fake = self.discriminator(torch.cat((rec_slices, cond), dim=1))

            # Hinge generator loss: G wants logits_fake to be high => -mean(logits_fake)
            g_loss = -torch.mean(logits_fake).to(torch.float)

            # Possibly compute adaptive weight for the adversarial term
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(unweighted_nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Final generator loss:
            loss = weighted_nll_loss + self.kl_weight * kl_loss \
                   + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss":       loss.detach().mean(),
                f"{split}/logvar":           self.logvar.detach(),
                f"{split}/kl_loss":          kl_loss.detach().mean(),
                f"{split}/nll_loss":         unweighted_nll_loss.detach().mean(),  # to match 2D logging
                f"{split}/rec_loss":         rec_loss.mean().detach(),             # L1+LPIPS mean
                f"{split}/d_weight":         d_weight.detach(),
                f"{split}/disc_factor":      torch.tensor(disc_factor),
                f"{split}/g_loss":           g_loss.detach().mean(),
            }
            return loss, log

        # -------------------------------------------------
        # 5) If Discriminator Update (optimizer_idx=1)
        # -------------------------------------------------
        elif optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(in_slices.detach())
                logits_fake = self.discriminator(rec_slices.detach())
            else:
                logits_real = self.discriminator(torch.cat((in_slices.detach(), cond.detach()), dim=1))
                logits_fake = self.discriminator(torch.cat((rec_slices.detach(), cond.detach()), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Hinge or vanilla disc loss
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss":   d_loss.detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log
        

class LPIPSWithDiscriminator3D_v5(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 n_slices=96):
        """
        A 3D loss module that:
          - Slices each [BS, C, D, D, D] volume into 2D planes (XY, XZ, YZ)
          - Computes L1 + LPIPS on those slices (as in the 2D version)
          - Includes a 2D NLayerDiscriminator for adversarial training on slices
          - Uses the same logvar weighting & kl_weight logic as LPIPSWithDiscriminator

        Args:
          disc_start (int): iteration at which to start adversarial training
          logvar_init (float): initial log-variance for NLL weighting
          kl_weight (float): how much to scale the KL term
          pixelloss_weight (float): optional multiplier for pixel (L1) term
          disc_num_layers (int): how many layers in the NLayerDiscriminator
          disc_in_channels (int): input channels for discriminator (3 if rgb, 1 if grayscale)
          disc_factor (float): overall multiplier for adversarial loss
          disc_weight (float): factor for the adaptive weight
          perceptual_weight (float): weight for the LPIPS term
          use_actnorm (bool): use ActNorm in the discriminator instead of BatchNorm
          disc_conditional (bool): if True, pass `cond` into discriminator as extra channels
          disc_loss (str): "hinge" or "vanilla" (BCE) for the adversarial criterion
          n_slices (int): how many slices to sample per orientation (XY, XZ, YZ)
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # -- Main weights and hyperparams --
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        # -- Slicing config --
        self.n_slices = n_slices
        print(f"Number of slices for 3D volume: {n_slices}")

        # -- LPIPS (2D) --
        #   Use .eval() so it doesn’t update internal stats
        self.perceptual_loss = LPIPS().eval()

        # -- Learnable log-variance for nll_loss
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # -- 2D Discriminator & related config --
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term
        based on ratio of reconstruction grads to generator grads.
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

    def extract_subset_slices_from3d(self, in_slices, rec_slices, n_slices=16):
        """
        Randomly pick n_slices from the total slices along a given orientation.
        """
        total_slices = in_slices.shape[0]
        indices = torch.randperm(total_slices, device=in_slices.device)[:n_slices]

        in_slices = in_slices[indices]
        rec_slices = rec_slices[indices]
        return in_slices, rec_slices

    def to_slices_subset(self, inputs, reconstructions, n_slices=64):
        """
        Convert 3D volumes [BS, C, D, D, D] into a subset of 2D slices for XY, XZ, YZ.

        Returns:
          in_slices:  [N, C, H, W]
          rec_slices: [N, C, H, W]
        where N = n_slices * 3 for each batch entry (roughly).
        """
        BS, C, D, D2, D3 = inputs.shape
        assert (D == D2 == D3), "Expected cubic volume [BS, C, D, D, D]."

        # 1) XY slices (z in [0..D-1])
        #    shape => [BS*D, C, D, D]
        xy_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D, C, D, D)
        xy_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D, C, D, D)
        xy_in, xy_rec = self.extract_subset_slices_from3d(xy_in, xy_rec, n_slices=n_slices)

        # 2) XZ slices (w in [0..D-1])
        #    shape => [BS*D, C, D, D]
        xz_in = inputs.permute(0, 3, 1, 2, 4).reshape(BS * D, C, D, D)
        xz_rec = reconstructions.permute(0, 3, 1, 2, 4).reshape(BS * D, C, D, D)
        xz_in, xz_rec = self.extract_subset_slices_from3d(xz_in, xz_rec, n_slices=n_slices)

        # 3) YZ slices (h in [0..D-1])
        #    shape => [BS*D, C, D, D]
        yz_in = inputs.permute(0, 2, 1, 3, 4).reshape(BS * D, C, D, D)
        yz_rec = reconstructions.permute(0, 2, 1, 3, 4).reshape(BS * D, C, D, D)
        yz_in, yz_rec = self.extract_subset_slices_from3d(yz_in, yz_rec, n_slices=n_slices)

        # Concatenate along batch dim => [N, C, H, W]
        in_slices = torch.cat([xy_in, xz_in, yz_in], dim=0)
        rec_slices = torch.cat([xy_rec, xz_rec, yz_rec], dim=0)

        return in_slices, rec_slices

    def forward(self,
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train",
                weights=None):
        """
        Args:
          inputs:          [BS, C, D, D, D]
          reconstructions: [BS, C, D, D, D]
          posteriors:      object with .kl() method => KL of latent distribution
          optimizer_idx:   0 => generator step, 1 => discriminator step
          global_step:     current training step (for disc_factor scheduling)
          last_layer:      optional last layer for adaptive weight
          cond:            optional 2D conditioning appended as extra channels
          split:           'train' or 'val'
          weights:         optional weighting map

        Returns:
          (loss, log_dict) for whichever optimizer_idx is active.
        """

        # -------------------------------------------------
        # 1) Extract 2D slices for L1 + LPIPS
        # -------------------------------------------------
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions,
                                                      n_slices=self.n_slices)
        # rec_slices shape => [N, C, H, W], typically N = 3 * BS * n_slices

        # Pixel-level L1, then reduce over (C,H,W)
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous())
        # shape => [N,1,1,1]

        # If using LPIPS:
        if self.perceptual_weight > 0:
            # spatial_avg=False for a voxel-wise loss value:
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            # here we have a voxel-wise loss value for reconstruction  L1+LPIPS
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # -------------------------------------------------
        # 2) Combine into negative log-likelihood with logvar
        #    nll_loss = rec_loss / exp(logvar) + logvar
        # -------------------------------------------------
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        # shape => [N,1,1,1]

        # Optional voxel/ slice weighting
        if weights is not None:
            # ensure weights has shape [N,1,1,1] or broadcasts
            nll_loss = nll_loss * weights

        # For logging, store the unweighted average, as in 2D:
        # unweighted_nll_loss = nll_loss.mean()  # overall mean

        # We'll do the final "weighted_nll_loss" as well:
        weighted_nll_loss = nll_loss.sum() / nll_loss.shape[0]  
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # -------------------------------------------------
        # 3) KL Loss
        # -------------------------------------------------
        kl = posteriors.kl()
        kl_loss = torch.sum(kl) / kl.shape[0]

        # -------------------------------------------------
        # 4) If Generator Update (optimizer_idx=0)
        # -------------------------------------------------
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional, "cond=None but disc_conditional=True?"
                logits_fake = self.discriminator(rec_slices.contiguous())
            else:
                assert self.disc_conditional, "cond is provided but disc_conditional=False?"
                logits_fake = self.discriminator(torch.cat((rec_slices.contiguous(), cond), dim=1))

            # Hinge generator loss: G wants logits_fake to be high => -mean(logits_fake)
            g_loss = -torch.mean(logits_fake).to(torch.float)

            # Possibly compute adaptive weight for the adversarial term
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Final generator loss:
            loss = weighted_nll_loss + self.kl_weight * kl_loss \
                   + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss":       loss.clone().detach().mean(),
                f"{split}/logvar":           self.logvar.detach(),
                f"{split}/kl_loss":          kl_loss.detach().mean(),
                f"{split}/nll_loss":         nll_loss.detach().mean(),  # to match 2D logging
                f"{split}/rec_loss":         rec_loss.mean().detach(),             # L1+LPIPS mean
                f"{split}/d_weight":         d_weight.detach(),
                f"{split}/disc_factor":      torch.tensor(disc_factor),
                f"{split}/g_loss":           g_loss.detach().mean(),
            }
            return loss, log

        # -------------------------------------------------
        # 5) If Discriminator Update (optimizer_idx=1)
        # -------------------------------------------------
        elif optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(in_slices.contiguous().detach())
                logits_fake = self.discriminator(rec_slices.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((in_slices.contiguous().detach(), cond.contiguous().detach()), dim=1))
                logits_fake = self.discriminator(torch.cat((rec_slices.contiguous().detach(), cond.contiguous().detach()), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Hinge or vanilla disc loss
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss":   d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log
        
class LPIPSWithDiscriminator3D_v6(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=3,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 n_slices=96):
        """
        A 3D loss module that:
          - Slices each [BS, C, D, D2, D3] volume into 2D planes (XY, XZ, YZ)
          - Computes L1 + LPIPS on those slices (as in the 2D version)
          - Includes a 2D NLayerDiscriminator for adversarial training on slices
          - Uses the same logvar weighting & kl_weight logic as LPIPSWithDiscriminator

        Args:
          disc_start (int): iteration at which to start adversarial training
          logvar_init (float): initial log-variance for NLL weighting
          kl_weight (float): how much to scale the KL term
          pixelloss_weight (float): optional multiplier for pixel (L1) term
          disc_num_layers (int): how many layers in the NLayerDiscriminator
          disc_in_channels (int): input channels for discriminator (3 if rgb, 1 if grayscale)
          disc_factor (float): overall multiplier for adversarial loss
          disc_weight (float): factor for the adaptive weight
          perceptual_weight (float): weight for the LPIPS term
          use_actnorm (bool): use ActNorm in the discriminator instead of BatchNorm
          disc_conditional (bool): if True, pass `cond` into discriminator as extra channels
          disc_loss (str): "hinge" or "vanilla" (BCE) for the adversarial criterion
          n_slices (int): how many slices to sample per orientation (XY, XZ, YZ)
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # -- Main weights and hyperparams --
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        # -- Slicing config --
        self.n_slices = n_slices
        print(f"Number of slices for 3D volume: {n_slices}")

        # -- LPIPS (2D) --
        #   Use .eval() so it doesn’t update internal stats
        self.perceptual_loss = LPIPS().eval()

        # -- Learnable log-variance for nll_loss
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # -- 2D Discriminator & related config --
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term
        based on ratio of reconstruction grads to generator grads.
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

    def extract_subset_slices_from3d(self, in_slices, rec_slices, n_slices=16):
        """
        Randomly pick n_slices from the total slices along a given orientation.
        """
        total_slices = in_slices.shape[0]
        indices = torch.randperm(total_slices, device=in_slices.device)[:n_slices]

        in_slices = in_slices[indices]
        rec_slices = rec_slices[indices]
        return in_slices, rec_slices

    def to_slices_subset(self, inputs, reconstructions, n_slices=64):
        """
        Convert 3D volumes [BS, C, D, D2, D3] into a subset of 2D slices for XY, XZ, YZ.

        Returns:
          in_slices:  [N, C, H, W]
          rec_slices: [N, C, H, W]
        where N = n_slices * 3 for each batch entry (roughly).
        """

        BS, C, D, D2, D3 = inputs.shape
        # 1) XY slices (z in [0..D-1])
        #    D3 is the smallest dimension of the patch
        xy_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        xy_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        in_slices = xy_in
        rec_slices = xy_rec

        return in_slices, rec_slices

    def forward(self,
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train",
                weights=None):
        """
        Args:
          inputs:          [BS, C, D, D2, D3] D=D2 > D3
          reconstructions: [BS, C, D, D2, D3]
          posteriors:      object with .kl() method => KL of latent distribution
          optimizer_idx:   0 => generator step, 1 => discriminator step
          global_step:     current training step (for disc_factor scheduling)
          last_layer:      optional last layer for adaptive weight
          cond:            optional 2D conditioning appended as extra channels
          split:           'train' or 'val'
          weights:         optional weighting map

        Returns:
          (loss, log_dict) for whichever optimizer_idx is active.
        """

        # -------------------------------------------------
        # 1) Extract 2D slices for L1 + LPIPS
        # -------------------------------------------------
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions,
                                                      n_slices=self.n_slices)
        # rec_slices shape => [N, C, H, W], typically N = 3 * BS * n_slices

        # Pixel-level L1, then reduce over (C,H,W)
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous())
        # shape => [N,1,1,1]

        # If using LPIPS:
        if self.perceptual_weight > 0:
            # spatial_avg=False for a voxel-wise loss value:
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            # here we have a voxel-wise loss value for reconstruction  L1+LPIPS
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # -------------------------------------------------
        # 2) Combine into negative log-likelihood with logvar
        #    nll_loss = rec_loss / exp(logvar) + logvar
        # -------------------------------------------------
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        # shape => [N,1,1,1]

        # Optional voxel/ slice weighting
        if weights is not None:
            # ensure weights has shape [N,1,1,1] or broadcasts
            nll_loss = nll_loss * weights

        # For logging, store the unweighted average, as in 2D:
        # unweighted_nll_loss = nll_loss.mean()  # overall mean

        # We'll do the final "weighted_nll_loss" as well:
        weighted_nll_loss = nll_loss.sum() / nll_loss.shape[0]  
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        # -------------------------------------------------
        # 3) KL Loss
        # -------------------------------------------------
        kl = posteriors.kl()
        kl_loss = torch.sum(kl) / kl.shape[0]

        # -------------------------------------------------
        # 4) If Generator Update (optimizer_idx=0)
        # -------------------------------------------------
        if optimizer_idx == 0:
            if cond is None:
                assert not self.disc_conditional, "cond=None but disc_conditional=True?"
                logits_fake = self.discriminator(rec_slices.contiguous())
            else:
                assert self.disc_conditional, "cond is provided but disc_conditional=False?"
                logits_fake = self.discriminator(torch.cat((rec_slices.contiguous(), cond), dim=1))

            # Hinge generator loss: G wants logits_fake to be high => -mean(logits_fake)
            g_loss = -torch.mean(logits_fake).to(torch.float)

            # Possibly compute adaptive weight for the adversarial term
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Final generator loss:
            loss = weighted_nll_loss + self.kl_weight * kl_loss \
                   + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss":       loss.clone().detach().mean(),
                f"{split}/logvar":           self.logvar.detach(),
                f"{split}/kl_loss":          kl_loss.detach().mean(),
                f"{split}/nll_loss":         nll_loss.detach().mean(),  # to match 2D logging
                f"{split}/rec_loss":         rec_loss.mean().detach(),             # L1+LPIPS mean
                f"{split}/d_weight":         d_weight.detach(),
                f"{split}/disc_factor":      torch.tensor(disc_factor),
                f"{split}/g_loss":           g_loss.detach().mean(),
            }
            return loss, log

        # -------------------------------------------------
        # 5) If Discriminator Update (optimizer_idx=1)
        # -------------------------------------------------
        elif optimizer_idx == 1:
            if cond is None:
                logits_real = self.discriminator(in_slices.contiguous().detach())
                logits_fake = self.discriminator(rec_slices.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((in_slices.contiguous().detach(), cond.contiguous().detach()), dim=1))
                logits_fake = self.discriminator(torch.cat((rec_slices.contiguous().detach(), cond.contiguous().detach()), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            # Hinge or vanilla disc loss
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss":   d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log

class LPIPSWithDiscriminator3D_v7(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=1.0,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=32,
                 disc_rampup_length=5000):
        """
        A 3D loss module that:
          - Uses full 3D discriminator on volumes
          - Scales slice-based losses for consistency
          - Uses regularization to prevent logit explosion
          - Provides gradual discriminator ramp-up
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # -- Main weights and hyperparams --
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.ndf = num_channels

        # -- LPIPS (2D) --
        self.perceptual_loss = LPIPS().eval()

        # -- Learnable log-variance for nll_loss - CHANGED FOR FSDP COMPATIBILITY
        self.logvar = nn.Parameter(torch.ones(size=(1,)) * logvar_init)

        # -- 3D Discriminator --
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        # -- Training config --
        self.discriminator_iter_start = disc_start
        self.disc_rampup_length = disc_rampup_length
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
        # -- Regularization --
        self.logit_reg_weight = 0.1  # Weight for direct logit magnitude regularization

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term
        based on ratio of reconstruction grads to generator grads.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            raise ValueError("last_layer should be passed if using adaptive weight.")
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def to_slices_subset(self, inputs, reconstructions):
        """
        Extract all xy slices for perceptual loss calculation
        """
        BS, C, D, D2, D3 = inputs.shape
        # XY slices (z in [0..D3-1])
        xy_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        xy_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        return xy_in, xy_rec

    def forward(self,
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train",
                weights=None):
        """
        Modified forward pass with proper rescaling and 3D discriminator
        """
        # -------------------------------------------------
        # 1) Extract 2D slices for L1 + LPIPS
        # -------------------------------------------------
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions)
        
        # Compute scaling factor to normalize losses based on slice count
        original_batch_size = inputs.shape[0]
        effective_batch_size = in_slices.shape[0]
        scaling_factor = original_batch_size / effective_batch_size

        # Pixel-level L1
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous())

        # Perceptual loss via LPIPS on 2D slices
        if self.perceptual_weight > 0:
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # -------------------------------------------------
        # 2) Combine into negative log-likelihood with logvar scaling
        # -------------------------------------------------
        nll_loss = rec_loss / torch.exp(self.logvar[0]) + self.logvar[0]  # FSDP fix
        weighted_nll_loss = nll_loss
        
        if weights is not None:
            weighted_nll_loss = weighted_nll_loss * weights

        # Apply scaling factor
        weighted_nll_loss = (weighted_nll_loss.sum() / weighted_nll_loss.shape[0]) * scaling_factor
        nll_loss = (torch.sum(nll_loss) / nll_loss.shape[0]) * scaling_factor
        
        # -------------------------------------------------
        # 3) KL Loss
        # -------------------------------------------------
        kl = posteriors.kl()
        kl_loss = torch.sum(kl) / kl.shape[0]

        # -------------------------------------------------
        # 4) If Generator Update (optimizer_idx=0)
        # -------------------------------------------------
        if optimizer_idx == 0:
            # Add instance noise that decreases over time
            noise_scale = max(0, 0.05 * (1 - global_step/20000))
            if noise_scale > 0:
                noisy_reconstructions = reconstructions + noise_scale * torch.randn_like(reconstructions)
            else:
                noisy_reconstructions = reconstructions
                
            # Use full 3D volumes with 3D discriminator
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(noisy_reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((noisy_reconstructions.contiguous(), cond), dim=1))

            # Generator loss (wants discriminator to think outputs are real)
            g_loss = -torch.mean(logits_fake).to(torch.float)

            # Compute adaptive weight
            if self.disc_factor > 0.0 and split == "train":
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            # Apply discriminator scheduling with gradual ramp-up
            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if global_step > self.discriminator_iter_start:
                rampup_factor = min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
            else:
                rampup_factor = 0.0                
            disc_factor = base_disc_factor * rampup_factor

            # Final generator loss
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {
                f"{split}/total_loss":      loss.clone().detach().mean(),
                f"{split}/logvar":          self.logvar.detach(),  # Still works with 1D tensor
                f"{split}/kl_loss":         kl_loss.detach().mean(),
                f"{split}/nll_loss":        nll_loss.detach().mean(), 
                f"{split}/rec_loss":        rec_loss.mean().detach() * scaling_factor,
                f"{split}/d_weight":        d_weight.detach(),
                f"{split}/disc_factor":     torch.tensor(disc_factor),
                f"{split}/rampup_factor":   torch.tensor(rampup_factor),
                f"{split}/g_loss":          g_loss.detach().mean(),
                f"{split}/noise_scale":     torch.tensor(noise_scale),
                f"{split}/scaling_factor":  torch.tensor(scaling_factor),
            }
            return loss, log

        # -------------------------------------------------
        # 5) If Discriminator Update (optimizer_idx=1)
        # -------------------------------------------------
        elif optimizer_idx == 1:
            # Add noise to both real and fake inputs
            noise_scale = max(0, 0.05 * (1 - global_step/20000))
            if noise_scale > 0:
                noisy_inputs = inputs + noise_scale * torch.randn_like(inputs)
                noisy_reconstructions = reconstructions + noise_scale * torch.randn_like(reconstructions)
            else:
                noisy_inputs = inputs
                noisy_reconstructions = reconstructions
            
            if cond is None:
                logits_real = self.discriminator(noisy_inputs.contiguous().detach())
                logits_fake = self.discriminator(noisy_reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((noisy_inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((noisy_reconstructions.contiguous().detach(), cond), dim=1))

            # Apply scheduling
            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if global_step > self.discriminator_iter_start:
                rampup_factor = min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
            else:
                rampup_factor = 0.0
            disc_factor = base_disc_factor * rampup_factor

            # Discriminator loss (standard)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)
            
            # Add direct logit magnitude regularization
            logit_regularization = self.logit_reg_weight * (
                torch.mean(torch.relu(torch.abs(logits_real) - 2.0)) + 
                torch.mean(torch.relu(torch.abs(logits_fake) - 2.0))
            )
            d_loss = d_loss + logit_regularization * disc_factor
            
            # -----------------------------------------------
            # Skip gradient penalty if not training
            # -----------------------------------------------
            if split == "train" and (global_step > self.discriminator_iter_start):
                # Only compute gradient penalty during training
                with torch.set_grad_enabled(True):  # Explicitly enable gradients
                    # Interpolate between real and fake
                    alpha = torch.rand(inputs.size(0), 1, 1, 1, 1, device=inputs.device)
                    interpolates = alpha * inputs + (1 - alpha) * reconstructions
                    interpolates.requires_grad_(True)
                    
                    disc_interpolates = self.discriminator(interpolates)
                    
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates, 
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(disc_interpolates),
                        create_graph=True, 
                        retain_graph=True
                    )[0]
                    
                    gradients = gradients.view(inputs.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
                    d_loss = d_loss + gradient_penalty
            else:
                gradient_penalty = torch.tensor(0.0, device=inputs.device)

            # Parameter L2 regularization
            l2_reg = 0.0001 * sum(torch.sum(p**2) for p in self.discriminator.parameters())
            d_loss = d_loss + l2_reg * disc_factor

            log = {
                f"{split}/disc_loss":         d_loss.clone().detach().mean(),
                f"{split}/logits_real":       logits_real.detach().mean(),
                f"{split}/logits_fake":       logits_fake.detach().mean(),
                f"{split}/disc_factor":       torch.tensor(disc_factor),
                f"{split}/rampup_factor":     torch.tensor(rampup_factor),
                f"{split}/gradient_penalty":  gradient_penalty.detach(),
                f"{split}/logit_reg":         logit_regularization.detach(),
                f"{split}/noise_scale":       torch.tensor(noise_scale),
                f"{split}/scaling_factor":    torch.tensor(scaling_factor),
            }
            return d_loss, log


class LPIPSWithDiscriminator3D_v8(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0e-08,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=1e-2,
                 perceptual_weight=1.0,
                 perceptual_on_cpu=True,  # New parameter: use CPU for perceptual loss by default
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=32,
                 disc_rampup_length=2): #Was 5000
        """
        A 3D loss module that:
          - Uses a full 3D discriminator on volumes.
          - Scales slice-based losses for consistency.
          - Uses regularization to prevent logit explosion.
          - Provides gradual discriminator ramp-up.
          - Optionally computes perceptual loss on CPU to reduce GPU memory usage.
          - Replicates 1-channel inputs to 3 channels for LPIPS.
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # Main weights and hyperparameters
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.ndf = num_channels
        self.perceptual_on_cpu = perceptual_on_cpu

        # 2D Perceptual loss (LPIPS)
        self.perceptual_loss = LPIPS().eval()
        # If using CPU for perceptual loss, move the module to CPU.
        if self.perceptual_on_cpu:
            self.perceptual_loss = self.perceptual_loss.to("cpu")

        # Learnable log-variance for nll_loss (for FSDP compatibility)
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # 3D Discriminator initialization
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        # Training configuration
        self.discriminator_iter_start = disc_start
        self.disc_rampup_length = disc_rampup_length
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        # Regularization: direct logit magnitude regularization weight
        self.logit_reg_weight = 0.1

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term based on the ratio
        of reconstruction gradients to generator gradients.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # Fallback: use self.last_layer[0] (make sure to define self.last_layer in your model)
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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", weights=None):
        """
        Forward pass that handles both generator and discriminator updates with proper rescaling.
        """
        # --- 1) Compute reconstruction losses on 2D slices ---
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions)
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous()) * self.pixel_weight

        if self.perceptual_weight > 0:
            if self.perceptual_on_cpu:
                # Move the slices to CPU for perceptual loss computation.
                in_slices_cpu = in_slices.detach().cpu()
                rec_slices_cpu = rec_slices.detach().cpu()
                # If single-channel, replicate to 3 channels.
                if in_slices_cpu.shape[1] == 1:
                    in_slices_cpu = in_slices_cpu.repeat(1, 3, 1, 1)
                    rec_slices_cpu = rec_slices_cpu.repeat(1, 3, 1, 1)
                # Ensure a CPU version of the perceptual loss module is used.
                lpips_cpu = self.perceptual_loss.cpu()
                p_loss_3d = lpips_cpu(in_slices_cpu, rec_slices_cpu, spatial_avg=False)
                # Move the computed loss back to the device of the inputs.
                p_loss_3d = p_loss_3d.to(inputs.device)
            else:
                # If not using CPU, ensure inputs have 3 channels.
                if in_slices.shape[1] == 1:
                    in_slices = in_slices.repeat(1, 3, 1, 1)
                    rec_slices = rec_slices.repeat(1, 3, 1, 1)
                p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # Negative log-likelihood loss.
        rec_loss = torch.mean(rec_loss)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        # --- 2) KL Divergence Loss ---
        kl_loss = torch.mean(posteriors.kl())

        # --- 3) Generator Update (optimizer_idx == 0) ---
        if optimizer_idx == 0:
            noise_scale = 0  # Currently no noise is applied.
            noisy_reconstructions = reconstructions

            if cond is None:
                assert not self.disc_conditional, "Discriminator is set to conditional but no condition was provided."
                logits_fake = self.discriminator(noisy_reconstructions.contiguous())
            else:
                assert self.disc_conditional, "Discriminator is not set to conditional but condition was provided."
                logits_fake = self.discriminator(torch.cat((noisy_reconstructions.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0 and split == "train":
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            rampup_factor = (min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
                             if global_step > self.discriminator_iter_start else 0.0)
            disc_factor = base_disc_factor * rampup_factor
            kl_loss = self.kl_weight * kl_loss
            disc_loss = d_weight * disc_factor * g_loss
            loss = nll_loss + kl_loss + disc_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/gan_g_loss": disc_loss.detach(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/logvar": self.logvar.detach(),
                
            }
            return loss, log

        # --- 4) Discriminator Update (optimizer_idx == 1) ---
        elif optimizer_idx == 1:
            noise_scale = 0  # Currently no noise is applied.
            if noise_scale > 0:
                noisy_inputs = inputs + noise_scale * torch.randn_like(inputs)
                noisy_reconstructions = reconstructions + noise_scale * torch.randn_like(reconstructions)
            else:
                noisy_inputs = inputs
                noisy_reconstructions = reconstructions

            if cond is None:
                logits_real = self.discriminator(noisy_inputs.contiguous().detach())
                logits_fake = self.discriminator(noisy_reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((noisy_inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((noisy_reconstructions.contiguous().detach(), cond), dim=1))

            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            rampup_factor = (min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
                             if global_step > self.discriminator_iter_start else 0.0)
            disc_factor = base_disc_factor * rampup_factor

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/gan_d_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/rampup_factor": torch.tensor(rampup_factor),
                f"{split}/gradient_penalty": gradient_penalty.detach()
            }
            return d_loss, log

class LPIPSWithDiscriminator3D_v9(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0e-8,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=1.0e-2,
                 perceptual_weight=1.0,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=32,
                 disc_rampup_length=1000):
        """
        A 3D loss module that:
          - Uses a full 3D discriminator on volumes
          - Scales slice-based losses for consistency
          - Uses regularization to prevent logit explosion
          - Provides gradual discriminator ramp-up
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # -- Main weights and hyperparams --
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.ndf = num_channels

        # -- LPIPS (2D) --
        self.perceptual_loss = LPIPS().eval()

        # -- Learnable log-variance for nll_loss --
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # -- 3D Discriminator --
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        # -- Training config --
        self.discriminator_iter_start = disc_start
        self.disc_rampup_length = disc_rampup_length
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        
        # -- Regularization --
        self.logit_reg_weight = 0.1  # Weight for direct logit magnitude regularization

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term
        based on ratio of reconstruction grads to generator grads.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            raise ValueError("last_layer should be passed if using adaptive weight.")

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def to_slices_subset(self, inputs, reconstructions):
        """
        Extract all xy slices for perceptual loss calculation.
        Only XY orientation is used here.
        """
        BS, C, D, D2, D3 = inputs.shape
        # XY slices (across z dimension)
        xy_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        xy_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        return xy_in, xy_rec

    def forward(self,
                inputs,
                reconstructions,
                posteriors,
                optimizer_idx,
                global_step,
                last_layer=None,
                cond=None,
                split="train",
                weights=None):
        """
        Forward pass with:
          - 2D slice-based reconstruction + perceptual loss
          - 3D discriminator for adversarial training
          - Optional gradient penalty & logit reg in the discriminator
        """
        # 1) Compute reconstruction losses from 2D slices
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions)
        rec_loss = torch.abs(in_slices - rec_slices) * self.pixel_weight

        # LPIPS on 2D slices (no spatial_avg, then reduce with mean)
        if self.perceptual_weight > 0:
            p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        rec_loss = torch.mean(rec_loss)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        # 2) KL Divergence
        kl_loss = torch.mean(posteriors.kl())

        # -------------------------------------------------------------------
        # Generator Update (optimizer_idx=0)
        # -------------------------------------------------------------------
        if optimizer_idx == 0:
            # No instance noise
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            # Generator wants logits_fake to be high (i.e. -logits_fake to be low)
            g_loss = -torch.mean(logits_fake).to(torch.float)

            # Adaptive weight for the adversarial term
            if self.disc_factor > 0.0 and split == "train":
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            # Ramp-up for disc factor
            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if global_step > self.discriminator_iter_start:
                rampup_factor = min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
            else:
                rampup_factor = 0.0
            disc_factor = base_disc_factor * rampup_factor

            # Combine all terms
            kl_term   = self.kl_weight * kl_loss
            disc_term = d_weight * disc_factor * g_loss
            loss = nll_loss + kl_term + disc_term

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/kl_loss":    kl_term.detach().mean(),
                f"{split}/rec_loss":   rec_loss.detach().mean(),
                f"{split}/nll_loss":   nll_loss.detach().mean(),
                f"{split}/gan_g_loss": disc_term.detach(),
                f"{split}/d_weight":   d_weight.detach(),
                f"{split}/g_loss":     g_loss.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor),
                f"{split}/logvar":     self.logvar.detach(),
            }
            return loss, log

        # -------------------------------------------------------------------
        # Discriminator Update (optimizer_idx=1)
        # -------------------------------------------------------------------
        elif optimizer_idx == 1:
            # No instance noise
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            # Ramp-up
            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            if global_step > self.discriminator_iter_start:
                rampup_factor = min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
            else:
                rampup_factor = 0.0
            disc_factor = base_disc_factor * rampup_factor

            # Basic D loss
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            # Logit magnitude regularization
            logit_regularization = self.logit_reg_weight * (
                torch.mean(torch.relu(torch.abs(logits_real) - 2.0)) + 
                torch.mean(torch.relu(torch.abs(logits_fake) - 2.0))
            )
            d_loss = d_loss + logit_regularization * disc_factor

            # Optional gradient penalty (only in train, after disc_start)
            if split == "train" and (global_step > self.discriminator_iter_start):
                with torch.set_grad_enabled(True):
                    alpha = torch.rand(inputs.size(0), 1, 1, 1, 1, device=inputs.device)
                    interpolates = alpha * inputs + (1 - alpha) * reconstructions
                    interpolates.requires_grad_(True)

                    disc_interpolates = self.discriminator(interpolates)
                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(disc_interpolates),
                        create_graph=True,
                        retain_graph=True
                    )[0]

                    gradients = gradients.view(inputs.size(0), -1)
                    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10.0
                    d_loss = d_loss + gradient_penalty
            else:
                gradient_penalty = torch.tensor(0.0, device=inputs.device)

            # Parameter L2 regularization
            l2_reg = 0.0001 * sum(torch.sum(p**2) for p in self.discriminator.parameters())
            d_loss = d_loss + l2_reg * disc_factor

            log = {
                f"{split}/disc_loss":        d_loss.clone().detach().mean(),
                f"{split}/logits_real":      logits_real.detach().mean(),
                f"{split}/logits_fake":      logits_fake.detach().mean(),
                f"{split}/disc_factor":      torch.tensor(disc_factor),
                f"{split}/rampup_factor":    torch.tensor(rampup_factor),
                f"{split}/gradient_penalty": gradient_penalty.detach(),
                f"{split}/logit_reg":        logit_regularization.detach(),
                f"{split}/l2_reg":           l2_reg.detach(),
            }
            return d_loss, log

class LPIPSWithDiscriminator3D_v8_2(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0e-08,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=0.5,
                 perceptual_weight=1.0,
                 perceptual_on_cpu=False,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=8,
                 disc_rampup_length=2,
                 adaptweight_clamp_to=1e4,
                 ):
        """
        A 3D loss module that:
          - Uses a full 3D discriminator on volumes.
          - Scales slice-based losses for consistency.
          - Uses regularization to prevent logit explosion.
          - Provides gradual discriminator ramp-up.
          - Optionally computes perceptual loss on CPU to reduce GPU memory usage.
          - Replicates 1-channel inputs to 3 channels for LPIPS.
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # Main weights and hyperparameters
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.ndf = num_channels
        self.perceptual_on_cpu = perceptual_on_cpu
        self.adaptweight_clamp_to = adaptweight_clamp_to
        # 2D Perceptual loss (LPIPS)
        self.perceptual_loss = LPIPS().eval()
        # If using CPU for perceptual loss, move the module to CPU.
        if self.perceptual_on_cpu:
            self.perceptual_loss = self.perceptual_loss.to("cpu")

        # Learnable log-variance for nll_loss (for FSDP compatibility)
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # 3D Discriminator initialization
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        # Training configuration
        self.discriminator_iter_start = disc_start
        self.disc_rampup_length = disc_rampup_length
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        # Regularization: direct logit magnitude regularization weight
        self.logit_reg_weight = 0.1

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term based on the ratio
        of reconstruction gradients to generator gradients.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # Fallback: use self.last_layer[0] (make sure to define self.last_layer in your model)
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptweight_clamp_to).detach()
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

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", weights=None):
        """
        Forward pass that handles both generator and discriminator updates with proper rescaling.
        """
        # --- 1) Compute reconstruction losses on 2D slices ---
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions)
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous()) * self.pixel_weight

        if self.perceptual_weight > 0:
            if self.perceptual_on_cpu:
                # Move the slices to CPU for perceptual loss computation.
                in_slices_cpu = in_slices.detach().cpu()
                rec_slices_cpu = rec_slices.detach().cpu()
                # If single-channel, replicate to 3 channels.
                if in_slices_cpu.shape[1] == 1:
                    in_slices_cpu = in_slices_cpu.repeat(1, 3, 1, 1)
                    rec_slices_cpu = rec_slices_cpu.repeat(1, 3, 1, 1)
                # Ensure a CPU version of the perceptual loss module is used.
                lpips_cpu = self.perceptual_loss.cpu()
                p_loss_3d = lpips_cpu(in_slices_cpu, rec_slices_cpu, spatial_avg=False)
                # Move the computed loss back to the device of the inputs.
                p_loss_3d = p_loss_3d.to(inputs.device)
            else:
                # If not using CPU, ensure inputs have 3 channels.
                if in_slices.shape[1] == 1:
                    in_slices = in_slices.repeat(1, 3, 1, 1)
                    rec_slices = rec_slices.repeat(1, 3, 1, 1)
                p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # Negative log-likelihood loss.
        rec_loss = torch.mean(rec_loss)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        # --- 2) KL Divergence Loss ---
        kl_loss = torch.mean(posteriors.kl())

        # --- 3) Generator Update (optimizer_idx == 0) ---
        if optimizer_idx == 0:
            noise_scale = 0  # Currently no noise is applied.
            noisy_reconstructions = reconstructions

            if cond is None:
                assert not self.disc_conditional, "Discriminator is set to conditional but no condition was provided."
                logits_fake = self.discriminator(noisy_reconstructions.contiguous())
            else:
                assert self.disc_conditional, "Discriminator is not set to conditional but condition was provided."
                logits_fake = self.discriminator(torch.cat((noisy_reconstructions.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0 and split == "train":
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            rampup_factor = (min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
                             if global_step > self.discriminator_iter_start else 0.0)
            disc_factor = base_disc_factor * rampup_factor
            kl_loss = self.kl_weight * kl_loss
            disc_loss = d_weight * disc_factor * g_loss
            loss = nll_loss + kl_loss + disc_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/gan_g_loss": disc_loss.detach(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=inputs.device),
                f"{split}/logvar": self.logvar.detach(),

            }
            return loss, log

        # --- 4) Discriminator Update (optimizer_idx == 1) ---
        elif optimizer_idx == 1:
            noise_scale = 0  # Currently no noise is applied.
            if noise_scale > 0:
                noisy_inputs = inputs + noise_scale * torch.randn_like(inputs)
                noisy_reconstructions = reconstructions + noise_scale * torch.randn_like(reconstructions)
            else:
                noisy_inputs = inputs
                noisy_reconstructions = reconstructions

            if cond is None:
                logits_real = self.discriminator(noisy_inputs.contiguous().detach())
                logits_fake = self.discriminator(noisy_reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((noisy_inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((noisy_reconstructions.contiguous().detach(), cond), dim=1))

            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            rampup_factor = (min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
                             if global_step > self.discriminator_iter_start else 0.0)
            disc_factor = base_disc_factor * rampup_factor

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/gan_d_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=inputs.device), # Specifies GPU
                f"{split}/rampup_factor": torch.tensor(rampup_factor, device=inputs.device) # Specifies GPU (in disc step)
            }
            return d_loss, log

class LPIPSWithDiscriminator3D_v8_2_aniso(nn.Module):
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0e-08,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=0.5,
                 perceptual_weight=1.0,
                 perceptual_on_cpu=False,
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=8,
                 disc_rampup_length=2,
                 adaptweight_clamp_to=1e4,
                 ):
        """
        A 3D loss module that:
          - Uses a full 3D discriminator on volumes.
          - Scales slice-based losses for consistency.
          - Uses regularization to prevent logit explosion.
          - Provides gradual discriminator ramp-up.
          - Optionally computes perceptual loss on CPU to reduce GPU memory usage.
          - Replicates 1-channel inputs to 3 channels for LPIPS.
        """
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"], "disc_loss must be 'hinge' or 'vanilla'"

        # Main weights and hyperparameters
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.ndf = num_channels
        self.perceptual_on_cpu = perceptual_on_cpu
        self.adaptweight_clamp_to = adaptweight_clamp_to
        # 2D Perceptual loss (LPIPS)
        self.perceptual_loss = LPIPS().eval()
        # If using CPU for perceptual loss, move the module to CPU.
        if self.perceptual_on_cpu:
            self.perceptual_loss = self.perceptual_loss.to("cpu")

        # Learnable log-variance for nll_loss (for FSDP compatibility)
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        # 3D Discriminator initialization
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        # Training configuration
        self.discriminator_iter_start = disc_start
        self.disc_rampup_length = disc_rampup_length
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

        # Regularization: direct logit magnitude regularization weight
        self.logit_reg_weight = 0.1

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        """
        Dynamically compute a weight for the adversarial term based on the ratio
        of reconstruction gradients to generator gradients.
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            # Fallback: use self.last_layer[0] (make sure to define self.last_layer in your model)
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptweight_clamp_to).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def extract_subset_slices_from3d(self, in_slices, rec_slices, n_slices = 16):
        # Get a random permutation of indices from 0 to 287, then take the first n_slices indices
        indices = torch.randperm(in_slices.shape[0])[:n_slices]

        # Use these indices to select slices from the tensor along the first dimension
        in_slices = in_slices[indices]
        rec_slices = rec_slices[indices]

        return in_slices, rec_slices

    def to_slices_subset(self, inputs, reconstructions):
        """
        Extract all xy slices for perceptual loss calculation.
        Only XY orientation is used here.
        """
        BS, C, D, D2, D3 = inputs.shape
        # XY slices (across z dimension)
        xy_in = inputs.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        xy_rec = reconstructions.permute(0, 4, 1, 2, 3).reshape(BS * D3, C, D, D2)
        return xy_in, xy_rec

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", weights=None):
        """
        Forward pass that handles both generator and discriminator updates with proper rescaling.
        """
        # --- 1) Compute reconstruction losses on 2D slices ---
        in_slices, rec_slices = self.to_slices_subset(inputs, reconstructions)
        rec_loss = torch.abs(in_slices.contiguous() - rec_slices.contiguous()) * self.pixel_weight

        if self.perceptual_weight > 0:
            if self.perceptual_on_cpu:
                # Move the slices to CPU for perceptual loss computation.
                in_slices_cpu = in_slices.detach().cpu()
                rec_slices_cpu = rec_slices.detach().cpu()
                # If single-channel, replicate to 3 channels.
                if in_slices_cpu.shape[1] == 1:
                    in_slices_cpu = in_slices_cpu.repeat(1, 3, 1, 1)
                    rec_slices_cpu = rec_slices_cpu.repeat(1, 3, 1, 1)
                # Ensure a CPU version of the perceptual loss module is used.
                lpips_cpu = self.perceptual_loss.cpu()
                p_loss_3d = lpips_cpu(in_slices_cpu, rec_slices_cpu, spatial_avg=False)
                # Move the computed loss back to the device of the inputs.
                p_loss_3d = p_loss_3d.to(inputs.device)
            else:
                # If not using CPU, ensure inputs have 3 channels.
                if in_slices.shape[1] == 1:
                    in_slices = in_slices.repeat(1, 3, 1, 1)
                    rec_slices = rec_slices.repeat(1, 3, 1, 1)
                p_loss_3d = self.perceptual_loss(in_slices, rec_slices, spatial_avg=False)
            rec_loss = rec_loss + self.perceptual_weight * p_loss_3d

        # Negative log-likelihood loss.
        rec_loss = torch.mean(rec_loss)
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar

        # --- 2) KL Divergence Loss ---
        kl_loss = torch.mean(posteriors.kl())

        # --- 3) Generator Update (optimizer_idx == 0) ---
        if optimizer_idx == 0:
            noise_scale = 0  # Currently no noise is applied.
            noisy_reconstructions = reconstructions

            if cond is None:
                assert not self.disc_conditional, "Discriminator is set to conditional but no condition was provided."
                logits_fake = self.discriminator(noisy_reconstructions.contiguous())
            else:
                assert self.disc_conditional, "Discriminator is not set to conditional but condition was provided."
                logits_fake = self.discriminator(torch.cat((noisy_reconstructions.contiguous(), cond), dim=1))

            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0 and split == "train":
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=inputs.device)
            else:
                d_weight = torch.tensor(0.0, device=inputs.device)

            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            rampup_factor = (min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
                             if global_step > self.discriminator_iter_start else 0.0)
            disc_factor = base_disc_factor * rampup_factor
            kl_loss = self.kl_weight * kl_loss
            disc_loss = d_weight * disc_factor * g_loss
            loss = nll_loss + kl_loss + disc_loss

            log = {
                f"{split}/total_loss": loss.clone().detach().mean(),
                f"{split}/kl_loss": kl_loss.detach().mean(),
                f"{split}/rec_loss": rec_loss.detach().mean(),
                f"{split}/nll_loss": nll_loss.detach().mean(),
                f"{split}/gan_g_loss": disc_loss.detach(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/g_loss": g_loss.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=inputs.device),
                f"{split}/logvar": self.logvar.detach(),

            }
            return loss, log

        # --- 4) Discriminator Update (optimizer_idx == 1) ---
        elif optimizer_idx == 1:
            noise_scale = 0  # Currently no noise is applied.
            if noise_scale > 0:
                noisy_inputs = inputs + noise_scale * torch.randn_like(inputs)
                noisy_reconstructions = reconstructions + noise_scale * torch.randn_like(reconstructions)
            else:
                noisy_inputs = inputs
                noisy_reconstructions = reconstructions

            if cond is None:
                logits_real = self.discriminator(noisy_inputs.contiguous().detach())
                logits_fake = self.discriminator(noisy_reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(
                    torch.cat((noisy_inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(
                    torch.cat((noisy_reconstructions.contiguous().detach(), cond), dim=1))

            base_disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            rampup_factor = (min(1.0, (global_step - self.discriminator_iter_start) / self.disc_rampup_length)
                             if global_step > self.discriminator_iter_start else 0.0)
            disc_factor = base_disc_factor * rampup_factor

            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/gan_d_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=inputs.device), # Specifies GPU
                f"{split}/rampup_factor": torch.tensor(rampup_factor, device=inputs.device) # Specifies GPU (in disc step)
            }
            return d_loss, log
        
class LPIPSWithDiscriminator3D_v8_3(nn.Module):
    """
    v8_3  ── memory‑optimised version
      * identical to v8_2_aniso on the axial plane
      * additionally evaluates coronal + sagittal planes
      * ≈ constant peak VRAM thanks to per‑plane streaming,
        autocast‑FP16, checkpointed LPIPS, and expand().
    """

    # ----------------- constructor identical to your draft -----------------
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0e-08,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=0.5,
                 perceptual_weight=1.0,
                 perceptual_on_cpu=False,
                 lpips_gpu_batch_size=16,   # default lower to be safe
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=8,
                 disc_rampup_length=2,
                 adaptweight_clamp_to=1e4,
                 planes=("xy", "xz", "yz"),  # NEW
                 use_amp=True,               # NEW
                 ):
        super().__init__()
        # ---------- unchanged hyper‑parameters ----------
        assert disc_loss in ("hinge", "vanilla")
        self.kl_weight          = kl_weight
        self.pixel_weight       = pixelloss_weight
        self.perceptual_weight  = perceptual_weight
        self.ndf                = num_channels
        self.perceptual_on_cpu  = perceptual_on_cpu
        self.lpips_gpu_batch_sz = lpips_gpu_batch_size
        self.adaptweight_clamp_to = adaptweight_clamp_to
        self.planes             = planes
        self.use_amp            = use_amp

        # ---------- LPIPS -----------
        self.lpips_model = LPIPS().eval()    # will be (re)moved to correct device on demand

        # ---------- other parts identical to v8_2_aniso ----------
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_rampup_length       = disc_rampup_length
        self.disc_loss_type           = disc_loss
        self.disc_factor_base         = disc_factor
        self.discriminator_weight     = disc_weight
        self.disc_conditional         = disc_conditional
        self.logit_reg_weight         = 0.1
        self.register_buffer("last_layer_dummy_param",
                             torch.zeros(1), persistent=False)
        self._last_layer_for_loss = None

    # ------------- helpers -------------------------------------------------
    def _lazy_slices(self, x3d: torch.Tensor, plane: str):
        """
        Returns a view with shape (N_slices, C, H, W) **without** copying.
        Works for BS=1 but also for larger batch sizes.
        """
        B, C, H, W, D = x3d.shape  # you guaranteed  H=W=192, D=32
        if plane == "xy":          # axial  (32 slices)
            return x3d.permute(0, 4, 1, 2, 3).reshape(B*D, C, H, W)
        if plane == "xz":          # coronal (192 slices)
            return x3d.permute(0, 3, 1, 2, 4).reshape(B*W, C, H, D)
        if plane == "yz":          # sagittal (192 slices)
            return x3d.permute(0, 2, 1, 3, 4).reshape(B*H, C, W, D)
        raise ValueError(f"Unknown plane={plane}")

    @staticmethod
    def _pixel_loss(a, b):
        return torch.mean(torch.abs(a - b))

    # ---------- LPIPS in chunks, fp16 + checkpoint -------------------------
    def _lpips_mean(self, a, b, device):
        """
        Memory‑friendly LPIPS:
          • channel‑expand with .expand (no copy)
          • autocast(fp16)
          • optional batch‑chunking
          • torch.utils.checkpoint to avoid storing activations
        """
        # channel‑expand to 3 without materialising three copies
        if a.shape[1] == 1:
            a = a.expand(-1, 3, -1, -1)
            b = b.expand(-1, 3, -1, -1)

        model = self.lpips_model
        processing_dev = torch.device("cpu") if self.perceptual_on_cpu else device
        model = model.to(processing_dev)

        a = a.to(processing_dev)
        b = b.to(processing_dev)

        bs = a.shape[0] if self.lpips_gpu_batch_sz is None else self.lpips_gpu_batch_sz
        total, n_elem = 0.0, 0

        for s in range(0, a.shape[0], bs):
            a_batch = a[s:s+bs]
            b_batch = b[s:s+bs]

            with torch.cuda.amp.autocast(self.use_amp and processing_dev.type == "cuda"):
                # checkpoint saves about 30‑35 % memory at cost of 1 extra forward
                out = torch.utils.checkpoint.checkpoint(
                    lambda x,y: model(x, y, spatial_avg=False),
                    a_batch, b_batch,
                    use_reentrant=False)

            total += out.sum()
            n_elem += out.numel()

            # free ASAP
            del a_batch, b_batch, out
            if processing_dev.type == "cuda":
                torch.cuda.empty_cache()

        mean_val = total / max(1, n_elem)
        return mean_val.to(device)

    # -------- adaptive weight helper same as before (verbatim) ------------
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        Compute dynamic GAN weight ‑‑ robust to many call patterns.

        • Accepts last_layer as None / torch.Tensor / Parameter / list(Parameter)
        • Returns 0.0 if no valid, trainable parameter is found
        """
        # ---------- normalise to list -----------------------------------------
        if last_layer is None:
            layers = []
        elif isinstance(last_layer, torch.Tensor):
            layers = [last_layer]
        elif isinstance(last_layer, (list, tuple)):
            layers = [l for l in last_layer]
        else:                 # unsupported type
            layers = []

        # If still empty, fall back to the dummy param you registered
        if len(layers) == 0:
            layers = [self.last_layer_dummy_param]

        # Pick the first layer that requires grad
        target_param = None
        for p in layers:
            if isinstance(p, torch.Tensor) and p.requires_grad:
                target_param = p
                break
        if target_param is None:   # nothing we can differentiate through
            return torch.tensor(0.0, device=nll_loss.device)

        # ---------- gradient ratio -------------------------------------------
        nll_grads = torch.autograd.grad(nll_loss, target_param, retain_graph=True, allow_unused=True)[0]
        g_grads   = torch.autograd.grad(g_loss , target_param, retain_graph=True, allow_unused=True)[0]

        if nll_grads is None or g_grads is None:
            return torch.tensor(0.0, device=nll_loss.device)

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptweight_clamp_to).detach()
        return d_weight * self.discriminator_weight

    # ----------------------------- forward ---------------------------------
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):

        device = inputs.device
        last_layer = last_layer if last_layer is not None else self.get_last_layer()

        # ------------------------------------------------------------------ #
        # 1.  RECONSTRUCTION LOSS  (pixel + LPIPS)  plane‑by‑plane streaming #
        # ------------------------------------------------------------------ #
        pixel_loss_total = 0.0
        lpips_loss_total = 0.0
        planes_done      = 0

        for plane in self.planes:
            # create views, **no copy**
            gt_slices  = self._lazy_slices(inputs        , plane)
            rec_slices = self._lazy_slices(reconstructions, plane)

            # pixel   (needs grad)  – keep on GPU
            pix_loss_plane = self._pixel_loss(gt_slices, rec_slices)
            pixel_loss_total += pix_loss_plane

            # LPIPS   (grad wrt rec_slices) – memory‑friendly
            if self.perceptual_weight > 0:
                lpips_plane = self._lpips_mean(gt_slices, rec_slices, device)
                lpips_loss_total += lpips_plane

            planes_done += 1

            # free slices explicitly
            del gt_slices, rec_slices, pix_loss_plane
            torch.cuda.empty_cache()

        pixel_loss_avg = pixel_loss_total / planes_done
        lpips_loss_avg = lpips_loss_total / planes_done if self.perceptual_weight > 0 else torch.tensor(0.0, device=device)

        rec_loss = self.pixel_weight * pixel_loss_avg + self.perceptual_weight * lpips_loss_avg
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        kl_loss  = torch.mean(posteriors.kl())

        # ------------------------------------------------------------------ #
        # 2. GENERATOR UPDATE ---------------------------------------------- #
        # ------------------------------------------------------------------ #
        if optimizer_idx == 0:
            self.discriminator.to(device)

            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            g_loss_raw = -torch.mean(logits_fake)

            # ramp
            ramp = 0.0
            if global_step >= self.discriminator_iter_start:
                if self.disc_rampup_length > 0:
                    ramp = min(1.0, (global_step - self.discriminator_iter_start) / float(self.disc_rampup_length))
                else:
                    ramp = 1.0
            disc_factor = self.disc_factor_base * ramp

            adapt_w = torch.tensor(0.0, device=device)
            if disc_factor > 0. and split == "train":
                adapt_w = self.calculate_adaptive_weight(nll_loss, g_loss_raw, last_layer)

            g_gan_term = adapt_w * disc_factor * g_loss_raw
            total_loss = nll_loss + self.kl_weight * kl_loss + g_gan_term

            log = {
                f"{split}/total": total_loss.detach(),
                f"{split}/pix":   pixel_loss_avg.detach(),
                f"{split}/lpips": lpips_loss_avg.detach(),
                f"{split}/nll":   nll_loss.detach(),
                f"{split}/kl":    (self.kl_weight*kl_loss).detach(),
                f"{split}/g_raw": g_loss_raw.detach(),
                f"{split}/g_gan": g_gan_term.detach(),
                f"{split}/adapt_w": adapt_w.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=device),
                f"{split}/logvar": self.logvar.detach()
            }
            return total_loss, log

        # ------------------------------------------------------------------ #
        # 3. DISCRIMINATOR UPDATE ------------------------------------------ #
        # ------------------------------------------------------------------ #
        elif optimizer_idx == 1:
            self.discriminator.to(device)

            if cond is None:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
            else:
                cond = cond.to(device)
                logits_real = self.discriminator(torch.cat((inputs.detach(), cond),  dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond), dim=1))

            ramp = 0.0
            if global_step >= self.discriminator_iter_start:
                if self.disc_rampup_length > 0:
                    ramp = min(1.0, (global_step - self.discriminator_iter_start) / float(self.disc_rampup_length))
                else:
                    ramp = 1.0
            disc_factor = self.disc_factor_base * ramp

            d_loss_fn = hinge_d_loss if self.disc_loss_type == "hinge" else vanilla_d_loss
            d_loss = disc_factor * d_loss_fn(logits_real, logits_fake)

            log = {
                f"{split}/d_loss": d_loss.detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=device),
                f"{split}/ramp": torch.tensor(ramp, device=device)
            }
            return d_loss, log
        
class LPIPSWithDiscriminator3D_v8_4(nn.Module):
    """
    v8_3  ── memory‑optimised version
      * identical to v8_2_aniso on the axial plane
      * additionally evaluates coronal + sagittal planes
      * ≈ constant peak VRAM thanks to per‑plane streaming,
        autocast‑FP16, checkpointed LPIPS, and expand().
    """

    # ----------------- constructor identical to your draft -----------------
    def __init__(self,
                 disc_start,
                 logvar_init=0.0,
                 kl_weight=1.0e-08,
                 pixelloss_weight=1.0,
                 disc_num_layers=3,
                 disc_in_channels=1,
                 disc_factor=1.0,
                 disc_weight=0.5,
                 perceptual_weight=1.0,
                 perceptual_on_cpu=False,
                 lpips_gpu_batch_size=16,   # default lower to be safe
                 use_actnorm=False,
                 disc_conditional=False,
                 disc_loss="hinge",
                 num_channels=8,
                 disc_rampup_length=2,
                 adaptweight_clamp_to=1e4,
                 planes=("xy", "xz", "yz"),  # NEW
                 use_amp=True,               # NEW
                 ):
        super().__init__()
        # ---------- unchanged hyper‑parameters ----------
        assert disc_loss in ("hinge", "vanilla")
        self.kl_weight          = kl_weight
        self.pixel_weight       = pixelloss_weight
        self.perceptual_weight  = perceptual_weight
        self.ndf                = num_channels
        self.perceptual_on_cpu  = perceptual_on_cpu
        self.lpips_gpu_batch_sz = lpips_gpu_batch_size
        self.adaptweight_clamp_to = adaptweight_clamp_to
        self.planes             = planes
        self.use_amp            = use_amp

        # ---------- LPIPS -----------
        self.lpips_model = LPIPS().eval()    # will be (re)moved to correct device on demand

        # ---------- other parts identical to v8_2_aniso ----------
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator3D(
            input_nc=disc_in_channels,
            ndf=self.ndf,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_rampup_length       = disc_rampup_length
        self.disc_loss_type           = disc_loss
        self.disc_factor_base         = disc_factor
        self.discriminator_weight     = disc_weight
        self.disc_conditional         = disc_conditional
        self.logit_reg_weight         = 0.1
        self.register_buffer("last_layer_dummy_param",
                             torch.zeros(1), persistent=False)
        self._last_layer_for_loss = None

    # ------------- helpers -------------------------------------------------
    def _lazy_slices(self, x3d: torch.Tensor, plane: str):
        """
        Returns a view with shape (N_slices, C, H, W) **without** copying.
        Works for BS=1 but also for larger batch sizes.
        """
        B, C, H, W, D = x3d.shape  # you guaranteed  H=W=192, D=32
        if plane == "xy":          # axial  (32 slices)
            return x3d.permute(0, 4, 1, 2, 3).reshape(B*D, C, H, W)
        if plane == "xz":          # coronal (192 slices)
            return x3d.permute(0, 3, 1, 2, 4).reshape(B*W, C, H, D)
        if plane == "yz":          # sagittal (192 slices)
            return x3d.permute(0, 2, 1, 3, 4).reshape(B*H, C, W, D)
        raise ValueError(f"Unknown plane={plane}")

    @staticmethod
    def _pixel_loss(a, b):
        return torch.mean(torch.abs(a - b))

    # ---------- LPIPS in chunks, fp16 + checkpoint -------------------------
    def _lpips_mean(self, a, b, device):
        """
        Memory‑friendly LPIPS:
          • channel‑expand with .expand (no copy)
          • autocast(fp16)
          • optional batch‑chunking
          • torch.utils.checkpoint to avoid storing activations
        """
        # channel‑expand to 3 without materialising three copies
        if a.shape[1] == 1:
            a = a.expand(-1, 3, -1, -1)
            b = b.expand(-1, 3, -1, -1)

        model = self.lpips_model
        processing_dev = torch.device("cpu") if self.perceptual_on_cpu else device
        model = model.to(processing_dev)

        a = a.to(processing_dev)
        b = b.to(processing_dev)

        bs = a.shape[0] if self.lpips_gpu_batch_sz is None else self.lpips_gpu_batch_sz
        total, n_elem = 0.0, 0

        for s in range(0, a.shape[0], bs):
            a_batch = a[s:s+bs]
            b_batch = b[s:s+bs]

            with torch.cuda.amp.autocast(self.use_amp and processing_dev.type == "cuda"):
                # checkpoint saves about 30‑35 % memory at cost of 1 extra forward
                out = torch.utils.checkpoint.checkpoint(
                    lambda x,y: model(x, y, spatial_avg=False),
                    a_batch, b_batch,
                    use_reentrant=False)

            total += out.sum()
            n_elem += out.numel()

            # free ASAP
            del a_batch, b_batch, out
            if processing_dev.type == "cuda":
                torch.cuda.empty_cache()

        mean_val = total / max(1, n_elem)
        return mean_val.to(device)

    # -------- adaptive weight helper same as before (verbatim) ------------
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        Compute dynamic GAN weight ‑‑ robust to many call patterns.

        • Accepts last_layer as None / torch.Tensor / Parameter / list(Parameter)
        • Returns 0.0 if no valid, trainable parameter is found
        """
        # ---------- normalise to list -----------------------------------------
        if last_layer is None:
            layers = []
        elif isinstance(last_layer, torch.Tensor):
            layers = [last_layer]
        elif isinstance(last_layer, (list, tuple)):
            layers = [l for l in last_layer]
        else:                 # unsupported type
            layers = []

        # If still empty, fall back to the dummy param you registered
        if len(layers) == 0:
            layers = [self.last_layer_dummy_param]

        # Pick the first layer that requires grad
        target_param = None
        for p in layers:
            if isinstance(p, torch.Tensor) and p.requires_grad:
                target_param = p
                break
        if target_param is None:   # nothing we can differentiate through
            return torch.tensor(0.0, device=nll_loss.device)

        # ---------- gradient ratio -------------------------------------------
        nll_grads = torch.autograd.grad(nll_loss, target_param, retain_graph=True, allow_unused=True)[0]
        g_grads   = torch.autograd.grad(g_loss , target_param, retain_graph=True, allow_unused=True)[0]

        if nll_grads is None or g_grads is None:
            return torch.tensor(0.0, device=nll_loss.device)

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, self.adaptweight_clamp_to).detach()
        return d_weight * self.discriminator_weight

    # ----------------------------- forward ---------------------------------
    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):

        device = inputs.device
        last_layer = last_layer if last_layer is not None else self.get_last_layer()

        # ------------------------------------------------------------------ #
        # 1.  RECONSTRUCTION LOSS  (pixel + LPIPS)  plane‑by‑plane streaming #
        # ------------------------------------------------------------------ #
        pixel_loss_total = 0.0
        lpips_loss_total = 0.0
        planes_done      = 0

        for plane in self.planes:
            # create views, **no copy**
            gt_slices  = self._lazy_slices(inputs        , plane)
            rec_slices = self._lazy_slices(reconstructions, plane)

            # pixel   (needs grad)  – keep on GPU
            pix_loss_plane = self._pixel_loss(gt_slices, rec_slices)
            pixel_loss_total += pix_loss_plane

            # LPIPS   (grad wrt rec_slices) – memory‑friendly
            if self.perceptual_weight > 0:
                lpips_plane = self._lpips_mean(gt_slices, rec_slices, device)
                lpips_loss_total += lpips_plane

            planes_done += 1

            # free slices explicitly
            del gt_slices, rec_slices, pix_loss_plane
            torch.cuda.empty_cache()

        pixel_loss_avg = pixel_loss_total / planes_done
        lpips_loss_avg = lpips_loss_total / planes_done if self.perceptual_weight > 0 else torch.tensor(0.0, device=device)

        rec_loss = self.pixel_weight * pixel_loss_avg + self.perceptual_weight * lpips_loss_avg
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        kl_loss  = torch.mean(posteriors.kl())

        # ------------------------------------------------------------------ #
        # 2. GENERATOR UPDATE ---------------------------------------------- #
        # ------------------------------------------------------------------ #
        if optimizer_idx == 0:
            self.discriminator.to(device)

            if cond is None:
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))

            g_loss_raw = -torch.mean(logits_fake)

            # ramp
            ramp = 0.0
            if global_step >= self.discriminator_iter_start:
                if self.disc_rampup_length > 0:
                    ramp = min(1.0, (global_step - self.discriminator_iter_start) / float(self.disc_rampup_length))
                else:
                    ramp = 1.0
            disc_factor = self.disc_factor_base * ramp

            adapt_w = torch.tensor(0.0, device=device)
            if disc_factor > 0. and split == "train":
                adapt_w = self.calculate_adaptive_weight(nll_loss, g_loss_raw, last_layer)

            g_gan_term = adapt_w * disc_factor * g_loss_raw
            total_loss = nll_loss + self.kl_weight * kl_loss + g_gan_term

            log = {
                f"{split}/total": total_loss.detach(),
                f"{split}/pix":   pixel_loss_avg.detach(),
                f"{split}/lpips": lpips_loss_avg.detach(),
                f"{split}/nll":   nll_loss.detach(),
                f"{split}/kl":    (self.kl_weight*kl_loss).detach(),
                f"{split}/g_raw": g_loss_raw.detach(),
                f"{split}/g_gan": g_gan_term.detach(),
                f"{split}/adapt_w": adapt_w.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=device),
                f"{split}/logvar": self.logvar.detach()
            }
            return total_loss, log

        # ------------------------------------------------------------------ #
        # 3. DISCRIMINATOR UPDATE ------------------------------------------ #
        # ------------------------------------------------------------------ #
        elif optimizer_idx == 1:
            self.discriminator.to(device)

            if cond is None:
                logits_real = self.discriminator(inputs.detach())
                logits_fake = self.discriminator(reconstructions.detach())
            else:
                cond = cond.to(device)
                logits_real = self.discriminator(torch.cat((inputs.detach(), cond),  dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.detach(), cond), dim=1))

            ramp = 0.0
            if global_step >= self.discriminator_iter_start:
                if self.disc_rampup_length > 0:
                    ramp = min(1.0, (global_step - self.discriminator_iter_start) / float(self.disc_rampup_length))
                else:
                    ramp = 1.0
            disc_factor = self.disc_factor_base * ramp

            d_loss_fn = hinge_d_loss if self.disc_loss_type == "hinge" else vanilla_d_loss
            d_loss = disc_factor * d_loss_fn(logits_real, logits_fake)

            log = {
                f"{split}/d_loss": d_loss.detach(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=device),
                f"{split}/ramp": torch.tensor(ramp, device=device)
            }
            return d_loss, log

class LPIPSWithDiscriminator3D_v8_4_consistency(nn.Module):
    """
    v8_3  ── memory‑optimised version
      * identical to v8_2_aniso on the axial plane
      * additionally evaluates coronal + sagittal planes
      * ≈ constant peak VRAM thanks to per‑plane streaming,
        autocast‑FP16, checkpointed LPIPS, and expand().
    """

    # ----------------- constructor identical to your draft -----------------
    def __init__(self,
                 logvar_init=0.0,
                 pixelloss_weight=1.0,
                 perceptual_weight=1.0,
                 perceptual_on_cpu=False,
                 lpips_gpu_batch_size=16,   # default lower to be safe
                 use_actnorm=False,
                 planes=("xy", "xz", "yz"),  # NEW
                 use_amp=True,               # NEW
                 ):
        super().__init__()
        # ---------- unchanged hyper‑parameters ----------
        self.pixel_weight       = pixelloss_weight
        self.perceptual_weight  = perceptual_weight
        self.perceptual_on_cpu  = perceptual_on_cpu
        self.lpips_gpu_batch_sz = lpips_gpu_batch_size
        self.planes             = planes
        self.use_amp            = use_amp

        # ---------- LPIPS -----------
        self.lpips_model = LPIPS().eval()
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    # ------------- helpers -------------------------------------------------
    def _lazy_slices(self, x3d: torch.Tensor, plane: str):
        """
        Returns a view with shape (N_slices, C, H, W) **without** copying.
        Works for BS=1 but also for larger batch sizes.
        """
        B, C, H, W, D = x3d.shape  # you guaranteed  H=W=192, D=32
        if plane == "xy":          # axial  (32 slices)
            return x3d.permute(0, 4, 1, 2, 3).reshape(B*D, C, H, W)
        if plane == "xz":          # coronal (192 slices)
            return x3d.permute(0, 3, 1, 2, 4).reshape(B*W, C, H, D)
        if plane == "yz":          # sagittal (192 slices)
            return x3d.permute(0, 2, 1, 3, 4).reshape(B*H, C, W, D)
        raise ValueError(f"Unknown plane={plane}")

    @staticmethod
    def _pixel_loss(a, b):
        return torch.mean(torch.abs(a - b))

    # ---------- LPIPS in chunks, fp16 + checkpoint -------------------------
    def _lpips_mean(self, a, b, device):
        """
        Memory‑friendly LPIPS:
          • channel‑expand with .expand (no copy)
          • autocast(fp16)
          • optional batch‑chunking
          • torch.utils.checkpoint to avoid storing activations
        """
        # channel‑expand to 3 without materialising three copies
        if a.shape[1] == 1:
            a = a.expand(-1, 3, -1, -1)
            b = b.expand(-1, 3, -1, -1)

        model = self.lpips_model
        processing_dev = torch.device("cpu") if self.perceptual_on_cpu else device
        model = model.to(processing_dev)

        a = a.to(processing_dev)
        b = b.to(processing_dev)

        bs = a.shape[0] if self.lpips_gpu_batch_sz is None else self.lpips_gpu_batch_sz
        total, n_elem = 0.0, 0

        for s in range(0, a.shape[0], bs):
            a_batch = a[s:s+bs]
            b_batch = b[s:s+bs]

            with torch.cuda.amp.autocast(self.use_amp and processing_dev.type == "cuda"):
                # checkpoint saves about 30‑35 % memory at cost of 1 extra forward
                out = torch.utils.checkpoint.checkpoint(
                    lambda x,y: model(x, y, spatial_avg=False),
                    a_batch, b_batch,
                    use_reentrant=False)

            total += out.sum()
            n_elem += out.numel()

            # free ASAP
            del a_batch, b_batch, out
            if processing_dev.type == "cuda":
                torch.cuda.empty_cache()

        mean_val = total / max(1, n_elem)
        return mean_val.to(device)

    # ----------------------------- forward ---------------------------------
    def forward(self, inputs, reconstructions, optimizer_idx,
                global_step, split="train"):

        device = inputs.device

        # ------------------------------------------------------------------ #
        # 1.  RECONSTRUCTION LOSS  (pixel + LPIPS)  plane‑by‑plane streaming #
        # ------------------------------------------------------------------ #
        pixel_loss_total = 0.0
        # lpips_loss_total = 0.0
        planes_done      = 0

        for plane in self.planes:
            # create views, **no copy**
            gt_slices  = self._lazy_slices(inputs        , plane)
            rec_slices = self._lazy_slices(reconstructions, plane)

            # pixel   (needs grad)  – keep on GPU
            pix_loss_plane = self._pixel_loss(gt_slices, rec_slices)
            pixel_loss_total += pix_loss_plane

            # # LPIPS   (grad wrt rec_slices) – memory‑friendly
            # if self.perceptual_weight > 0:
            #     lpips_plane = self._lpips_mean(gt_slices, rec_slices, device)
            #     lpips_loss_total += lpips_plane

            planes_done += 1

            # free slices explicitly
            del gt_slices, rec_slices, pix_loss_plane
            torch.cuda.empty_cache()

        pixel_loss_avg = pixel_loss_total / planes_done
        # lpips_loss_avg = lpips_loss_total / planes_done if self.perceptual_weight > 0 else torch.tensor(0.0, device=device)

        # rec_loss = self.pixel_weight * pixel_loss_avg + self.perceptual_weight * lpips_loss_avg
        rec_loss = self.pixel_weight * pixel_loss_avg 
        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar\

        # ------------------------------------------------------------------ #
        # 2. GENERATOR UPDATE ---------------------------------------------- #
        # ------------------------------------------------------------------ #
        if optimizer_idx == 0:
            total_loss = nll_loss

            log = {
                f"{split}/total_cons": total_loss.detach(),
                f"{split}/pix_cons":   pixel_loss_avg.detach(),
                # f"{split}/lpips_cons": lpips_loss_avg.detach(),
                f"{split}/nll_cons":   nll_loss.detach(),
            }
            return total_loss, log
        