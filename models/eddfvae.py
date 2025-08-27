import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

import einops
import functorch
from tqdm import trange

from vaetc.models.vae import VAE
from vaetc.network.imagequality import mse, cossim, ssim
from vaetc.models.utils import detach_dict

from .deepvae import Encoder, Decoder

import piq

LOG2PI = math.log(math.pi * 2)

class Dissimilarity(nn.Module):
    """ a function d(x,y) that satisfy:
        - ∀x,y, d(x,y) >= 0
        - x == y => d(x,y) == 0
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

class MSE(Dissimilarity):
    """ Mean Squared Error """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).square().view(x.shape[0], -1).mean(dim=1)

class MAE(Dissimilarity):
    """ Mean Absolute Error """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).abs().view(x.shape[0], -1).mean(dim=1)

class SSE(Dissimilarity):
    """ Sum of Squared Error """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).square().view(x.shape[0], -1).sum(dim=1)

class SAE(Dissimilarity):
    """ Sum of Absolute Error """
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (x - y).abs().view(x.shape[0], -1).sum(dim=1)

class CrossEntropy(Dissimilarity):
    """ Cross Entropy """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()

        self.eps = float(eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        xc = x.clamp(0.0, 1.0)
        yc = y.clamp(self.eps, 1.0 - self.eps)
        return (-(torch.xlogy(xc, yc) + torch.xlogy(1 - xc, 1 - yc))).view(x.shape[0], -1).mean(dim=1)

class MseCossimSsim(Dissimilarity):
    """ MSE-COS-SSIM """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return mse(x, y) + 1-cossim() + 1-ssim(x, y)

class SSIM(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - ssim(x, y)

class CosSim(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - cossim(x, y)

class IWSSIM(Dissimilarity):

    def __init__(self) -> None:
        super().__init__()

        self.preprocess = transforms.Resize(size=(161, 161))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        y = self.preprocess(y)
        return 1 - piq.information_weighted_ssim(x, y, data_range=1, reduction="none")

class VIF(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.vif_p(x, y, data_range=1, reduction="none")

class FSIM(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.fsim(x, y, data_range=1, reduction="none")

class SRSIM(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.srsim(x, y, data_range=1, reduction="none")

class GMSD(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.gmsd(x, y, data_range=1, reduction="none")

class VSI(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.vsi(x, y, data_range=1, reduction="none")

class DSS(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.dss(x, y, data_range=1, reduction="none")

class ContentScore(Dissimilarity):

    def __init__(self) -> None:
        super().__init__()

        self.func = piq.ContentLoss(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.func(x, y)

class StyleScore(Dissimilarity):

    def __init__(self) -> None:
        super().__init__()

        self.func = piq.StyleLoss(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.func(x, y)

class HaarPSI(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.haarpsi(x, y, data_range=1, reduction="none")

class MDSI(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.mdsi(x, y, data_range=1, reduction="none")

class MSGMSD(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.multi_scale_gmsd(x, y, data_range=1, reduction="none")

class MSSSIM(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1 - piq.multi_scale_ssim(x, y, data_range=1, reduction="none")

class LPIPS(Dissimilarity):

    def __init__(self) -> None:
        super().__init__()

        self.func = piq.LPIPS(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.func(x, y)

class PieAPP(Dissimilarity):

    def __init__(self) -> None:
        super().__init__()

        self.func = piq.PieAPP(reduction="none", data_range=1)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.func(x, y)

class DISTS(Dissimilarity):

    def __init__(self) -> None:
        super().__init__()

        self.func = piq.DISTS(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.func(x, y)

class TotalVariation(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return 1. - piq.total_variation(y, reduction="none") / 64

class BRISQUE(Dissimilarity):

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return piq.brisque(y, reduction="none", data_range=1) / 100

class FactorizedGaussianBN(nn.Module):

    def __init__(self, z_dim: int, batchnorm_momentum: float = 0.9, inplace: bool = True):

        super().__init__()
        
        self.momentum = float(batchnorm_momentum)
        self.running_mean = nn.Parameter(torch.randn(size=[z_dim]), requires_grad=False)
        self.running_logvar = nn.Parameter(torch.randn(size=[z_dim]), requires_grad=False)
        self.num_batches_tracked = nn.Parameter(torch.tensor(0), requires_grad=False)

    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        if self.training:

            mean_batch = mean.mean(dim=0)
            logvar_batch = -math.log(logvar.shape[0]) + logvar.logsumexp(dim=0)

            exmean = mean_batch
            exlogvar = torch.stack([logvar_batch, torch.clamp(mean.var(dim=0), min=1e-10).log()], dim=0).logsumexp(dim=0)
            
            if self.num_batches_tracked.item() == 0:
                self.running_mean.copy_(exmean.detach())
                self.running_logvar.copy_(exlogvar.detach())
            else:
                self.running_mean.copy_((exmean * (1. - self.momentum) + self.running_mean * self.momentum).detach())
                self.running_logvar.copy_((exlogvar * (1. - self.momentum) + self.running_logvar * self.momentum).detach())
        
            self.num_batches_tracked.copy_((self.num_batches_tracked + 1).detach())
        
        else:

            exmean = self.running_mean
            exlogvar = self.running_logvar

        mean = (mean - exmean[None,:]) * (-0.5 * exlogvar).exp()[None,:]
        logvar = logvar - exlogvar[None,:]

        return mean, logvar

class EDDFVAE(VAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        dissimilarity_name = hyperparameters.get("dissimilarity", "mse-cossim-ssim")
        if not isinstance(dissimilarity_name, list):
            dissimilarity_name = [dissimilarity_name]
        dissimilarity_dict = {
            "mse": MSE,
            "sse": SSE,
            "mae": MAE,
            "sae": SAE,
            "ce": CrossEntropy,
            "mse-cossim-ssim": MseCossimSsim,
            "ssim": SSIM,
            "cossim": CosSim,
            "iwssim": IWSSIM,
            "vif": VIF,
            "fsim": FSIM,
            "srsim": SRSIM,
            "gmsd": GMSD,
            "vsi": VSI,
            "dss": DSS,
            "content-score": ContentScore,
            "style-score": StyleScore,
            "haarpsi": HaarPSI,
            "mdsi": MDSI,
            "msgmsd": MSGMSD,
            "msssim": MSSSIM,
            "lpips": LPIPS,
            "pieapp": PieAPP,
            "dists": DISTS,
            "tv": TotalVariation,
            "brisque": BRISQUE,
        }
        self.dissimilarities = nn.ModuleList([
            dissimilarity_dict[name]()
            for name in dissimilarity_name
        ])

        hidden_channels = hyperparameters.get("hidden_channels", [128, 256, 512, 1024])
        hidden_elements = hyperparameters.get("hidden_elements", 256)
        self.enc_block = Encoder(hidden_channels=hidden_channels      , hidden_elements=hidden_elements, z_dim=self.z_dim)
        self.dec_block = Decoder(hidden_channels=hidden_channels[::-1], hidden_elements=hidden_elements, z_dim=self.z_dim)
        self.omit_fgbn = bool(hyperparameters.get("omit_fgbn", False))
        if not self.omit_fgbn:
            self.fgbn = FactorizedGaussianBN(z_dim=self.z_dim)

        self.autoencoding_loss = str(hyperparameters.get("autoencoding_loss", "log"))
        if self.autoencoding_loss not in ["log", "trainable-variance", "optimal-variance", "fixed-variance"]:
            raise NotImplementedError(f"{self.autoencoding_loss} not implemented")
        if self.autoencoding_loss in ["trainable-variance", "optimal-variance", "log"]:
            self.loggamma = nn.Parameter(torch.tensor(0.0), requires_grad=(self.autoencoding_loss == "trainable-variance"))
        elif self.autoencoding_loss == "fixed-variance":
            self.fixed_gamma = float(hyperparameters.get("fixed_gamma", 1.0))
            self.loggamma = nn.Parameter(torch.tensor(math.log(self.fixed_gamma)), requires_grad=False)

    def encode_gauss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.enc_block(x)
        if not self.omit_fgbn:
            mean, logvar = self.fgbn(mean, logvar)
        return mean, logvar

    def loss(self,
        x: torch.Tensor,
        z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor,
        x2: torch.Tensor,
        progress: Optional[float] = None
    ):

        batch_size = x.shape[0]
        num_pixels = x.numel() // batch_size

        # AE loss
        eps = 1e-7
        xc  = torch.clamp(x , min=eps, max=1 - eps)
        x2c = torch.clamp(x2, min=eps, max=1 - eps)
        dissim = [func(xc, x2c) for func in self.dissimilarities]
        dissim = torch.stack(dissim, dim=1).sum(dim=1)
        eps = 1e-40
        dissim = torch.clamp(dissim, min=eps)
        
        # losses
        if self.autoencoding_loss == "log":
            
            loggamma = dissim.mean().log() - math.log(num_pixels / 2)
            loss_ae = dissim.mean().log() * num_pixels / 2
            self.loggamma.copy_(loggamma.detach())

        elif self.autoencoding_loss == "trainable-variance":
            
            loggamma = self.loggamma
            loss_ae = dissim * self.loggamma.neg().exp() + self.loggamma * num_pixels / 2
            loss_ae = loss_ae.mean()

        elif self.autoencoding_loss == "optimal-variance":
            
            if self.training:
                loggamma = dissim.mean().log() - math.log(num_pixels / 2)
                clamped_min = -6
                loggamma = clamped_min + torch.log1p((loggamma - clamped_min).exp()) # softplus with min -6 (note that loggamma=logσ^2)
                self.loggamma.copy_(loggamma.detach())
            
            loggamma = self.loggamma
            
            loss_ae = dissim * self.loggamma.neg().exp() + self.loggamma * num_pixels / 2
            loss_ae = loss_ae.mean()
            
        elif self.autoencoding_loss == "fixed-variance":
            
            loggamma = torch.full_like(dissim[0], fill_value=math.log(self.fixed_gamma))
            self.loggamma.copy_(loggamma.detach())
            loss_ae = dissim * loggamma.neg().exp() + loggamma * num_pixels / 2
            loss_ae = loss_ae.mean()

        else:

            raise NotImplementedError()
        
        loss_reg = torch.mean(self.regularization_term(mean, logvar))
        
        # SGVB Estimator of ELBO
        neqzx_logpxz = dissim * loggamma.neg().exp() + (loggamma + LOG2PI) * num_pixels / 2
        neqzx_logpxz = neqzx_logpxz.view(batch_size, -1).sum(dim=1).mean()
        kl_qzx_px = loss_reg.detach()
        sgvb_estimator = -neqzx_logpxz - kl_qzx_px

        # Total loss
        loss = loss_ae + loss_reg

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "gamma": loggamma.exp(),
            "loggamma": loggamma,
            "elbo": sgvb_estimator,
            "dissim": dissim.mean(),
        })

    def sample_decoder(self, xc: torch.Tensor, num_samples_per_instance: int = 1, num_steps: int = 100000) -> torch.Tensor:
        """
        sample with Langevin Dynamics
        return tensor with shape (batch size, n, #channels, height, width)
        """

        invgamma = math.exp(-self.loggamma.item())

        def calc_grad(xc: torch.Tensor, x2c: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:

            tiled_batch = xc.ndim == 5
            
            if tiled_batch:
                n = xc.shape[1]
                xc = einops.rearrange(xc, "b n c h w -> (b n) c h w")
                x2c = einops.rearrange(x2c, "b n c h w -> (b n) c h w")

            def func(xc: torch.Tensor, x2c: torch.Tensor) -> torch.Tensor:
            
                d = [func(xc[None,...], x2c[None,...]) for func in self.dissimilarities]
                d = torch.stack(d, dim=-1).sum(dim=-1)
                d = torch.clamp(d, min=eps)
                d = d.squeeze(0)

                return -d * invgamma

            grad = functorch.vmap(functorch.grad(func))(xc, x2c)

            if tiled_batch:
                grad = einops.rearrange(grad, "(b n) c h w -> b n c h w", n=n)

            return grad.detach()

        def schedule(t: float, n: int = 10):
            return 0.1 / n * t ** -0.55

        xc_tiled = xc[:,None,...].tile(1, num_samples_per_instance, 1, 1, 1).detach()
        results = xc_tiled.detach() + torch.randn_like(xc_tiled)

        for t in trange(num_steps):

            at = schedule(1+t)
            grad = calc_grad(results, xc_tiled)
            noise = torch.randn_like(results)

            diff = at * grad + (2 * at) ** 0.5 * noise
            results += diff.detach()

        return results.detach()
