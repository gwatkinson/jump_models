from typing import Optional

# from torch.distributions import MultivariateNormal
import torch
from torch import Tensor

from src.modules.losses.base_losses import LossWithTemperature, RegWithTemperatureLoss


def calculate_rank(sim_matrix: Tensor) -> Tensor:
    with torch.no_grad():
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device).repeat(sim_matrix.shape[0], 1).t()

        # X to Y ranking
        row_order_x = sim_matrix.argsort(descending=True, dim=-1)
        x_to_y = (row_order_x == labels).nonzero(as_tuple=True)[1]

        # Y to X ranking
        row_order_y = sim_matrix.argsort(descending=True, dim=0).t()
        y_to_x = (row_order_y == labels).nonzero(as_tuple=True)[1]

        batch_size = sim_matrix.shape[0]

        results = {
            "x_to_y_top1": (x_to_y == 0).float().mean(),
            "x_to_y_top5": (x_to_y < 5).float().mean(),
            "x_to_y_top10": (x_to_y < 10).float().mean(),
            "x_to_y_mean_pos": 1 + x_to_y.float().mean(),
            "x_to_y_mean_pos_normed": (1 + x_to_y.float().mean()) / batch_size,
            "y_to_x_top1": (y_to_x == 0).float().mean(),
            "y_to_x_top5": (y_to_x < 5).float().mean(),
            "y_to_x_top10": (y_to_x < 10).float().mean(),
            "y_to_x_mean_pos": 1 + y_to_x.float().mean(),
            "y_to_x_mean_pos_normed": (1 + y_to_x.float().mean()) / batch_size,
        }

    return results


class InfoNCE(LossWithTemperature):
    def __init__(
        self,
        norm: bool = True,
        temperature: float = 0.5,
        return_rank: bool = False,
        eps: float = 1e-8,
        name: Optional[str] = None,
        **kwargs,
    ):
        # Access temperature as self.temperature.value
        super().__init__(temperature=temperature, **kwargs)
        self.norm = norm
        self.eps = eps
        self.name = name or "InfoNCE"
        self.return_rank = return_rank

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum("i,j->ij", z1_abs, z2_abs) + self.eps)

        sim_matrix = torch.exp(sim_matrix / self.temperature.value)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = -torch.log(loss).mean()

        if self.return_rank:
            results = calculate_rank(sim_matrix)
            return {"loss": loss, **results}

        return {"loss": loss}


class RegInfoNCE(RegWithTemperatureLoss):
    loss_fn = InfoNCE
    name = "RegInfoNCE"


class NTXent(LossWithTemperature):
    def __init__(
        self,
        norm: bool = True,
        temperature: float = 0.5,
        return_rank: bool = False,
        eps: float = 1e-8,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(temperature=temperature, **kwargs)
        self.norm = norm
        self.eps = eps
        self.name = name or "NTXent"
        self.return_rank = return_rank

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum("i,j->ij", z1_abs, z2_abs) + self.eps)

        sim_matrix = torch.exp(sim_matrix / self.temperature.value)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  # This is the difference from InfoNCE
        loss = -torch.log(loss).mean()

        if self.return_rank:
            results = calculate_rank(sim_matrix)
            return {"loss": loss, **results}

        return {"loss": loss}


class RegNTXent(RegWithTemperatureLoss):
    loss_fn = NTXent
    name = "RegNTXent"


# class KLDivergenceMultiplePositives(LossWithTemperature):
#     def __init__(
#         self,
#         norm: bool = True,
#         temperature: float = 0.5,
#         eps: float = 1e-8,
#         **kwargs,
#     ):
#         super().__init__(temperature=temperature, **kwargs)
#         self.norm = norm
#         self.eps = eps

#     def forward(self, z1, z2, **kwargs) -> Tensor:
#         """
#         :param z1: batchsize, metric dim*2
#         :param z2: batchsize*num_conformers, metric dim
#         """
#         batch_size, _ = z1.size()
#         _, metric_dim = z2.size()

#         z1 = z1.view(batch_size, 2, metric_dim)
#         z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

#         if self.norm:
#             z1 = F.normalize(z1, dim=2)
#             z2 = F.normalize(z2, dim=2)

#         z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
#         z1_vars = torch.exp(z1[:, 1, :])  # [batch_size, metric_dim]
#         z2_means = z2.mean(1)  # [batch_size, metric_dim]
#         z2_vars = z2.var(1) + 1e-6  # [batch_size, metric_dim]
#         try:
#             normal1 = MultivariateNormal(z1_means, torch.diag_embed(z1_vars))
#         except Exception:
#             print(z1_vars)
#         try:
#             normal2 = MultivariateNormal(z2_means, torch.diag_embed(z2_vars))
#         except Exception:
#             print(z2_vars)

#         kl_div = torch.distributions.kl_divergence(normal2, normal1)
#         loss = kl_div.mean()

#         return loss
