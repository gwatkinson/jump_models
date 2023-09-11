# Modified from https://github.com/HannesStark/3DInfomax/blob/master/commons/losses.py

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
        name: str = "InfoNCE",
        **kwargs,
    ):
        # Access temperature as self.temperature
        super().__init__(temperature=temperature, **kwargs)
        self.norm = norm
        self.eps = eps
        self.name = name
        self.return_rank = return_rank

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum("i,j->ij", z1_abs, z2_abs) + self.eps)

        sim_matrix = torch.exp(sim_matrix / self.temperature)
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

        sim_matrix = torch.exp(sim_matrix / self.temperature)
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
