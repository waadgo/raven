import torch
import numpy as np


class AbstractDistribution:
    def sample(self):
        raise NotImplementedError()

    def mode(self):
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    def __init__(self, value):
        self.value = value

    def sample(self):
        return self.value

    def mode(self):
        return self.value


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class DiagonalGaussianDistribution3D(object):
    def __init__(self, parameters, deterministic=False):
        """
        parameters: 5D tensor of shape (batch_size, 2 * C, H, W, Z)
                    where the channel dimension is twice the number of latent channels,
                    representing concatenated mean and log-variance.
        deterministic: If True, treat distribution as deterministic.
        """
        self.parameters = parameters
        # Split parameters into mean and logvar along channel dimension (dim=1)
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        # Compute standard deviation and variance
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            # If deterministic, override variance and std to zero
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device)

    def sample(self):
        """Sample from the diagonal Gaussian distribution."""
        if self.deterministic:
            return self.mean
        noise = torch.randn_like(self.mean)  # Generate noise matching the shape of mean
        return self.mean + self.std * noise

    def kl(self, other=None):
        """
        Compute the KL divergence. If `other` is provided, compute KL divergence
        between this distribution and `other`. If not, compute KL divergence
        to a standard normal distribution.
        """
        if self.deterministic:
            return torch.tensor(0., device=self.parameters.device)
        # Sum over channels and spatial dimensions: [1,2,3,4]
        dims = [1, 2, 3, 4]
        if other is None:
            return 0.5 * torch.sum(self.mean.pow(2) + self.var - 1.0 - self.logvar, dim=dims)
        else:
            return 0.5 * torch.sum(
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0 - self.logvar + other.logvar,
                dim=dims
            )

    def nll(self, sample, dims=[1,2,3,4]):
        """
        Compute the negative log-likelihood of a sample under this distribution.
        `dims` specify dimensions to sum over.
        """
        if self.deterministic:
            return torch.tensor(0., device=self.parameters.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + ((sample - self.mean) ** 2) / self.var,
            dim=dims
        )

    def mode(self):
        """Return the mode of the distribution, which is the mean for Gaussian."""
        return self.mean

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    source: https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/losses.py#L12
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
