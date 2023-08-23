# Inspired from https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/losses/contrastive_loss_with_temperature.py

from typing import Optional, Union

import torch
from torch import Tensor, nn


def nt_xent_loss(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    temperature: float = 1.0,
):
    out = torch.cat([embeddings_a, embeddings_b], dim=0)
    n_samples = out.shape[0]

    # Calculate cosine similarity
    sim = torch.mm(out, out.t().contiguous())
    sim = torch.exp(sim / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity
    pos = torch.exp(torch.sum(embeddings_a * embeddings_b, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / neg).mean()

    return loss


class NtXentLoss(nn.Module):
    def __init__(
        self,
        temperature: Union[float, nn.Parameter] = 1.0,
        temperature_min: Optional[float] = 0.0,
        temperature_max: Optional[float] = 100.0,
        requires_grad: bool = False,
    ):
        super().__init__()

        if not temperature_min and not temperature_max:
            raise ValueError("Only one of `temperature_min` and `temperature_max` can be None.")
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max

        # If already initialized, set to what was passed
        if isinstance(temperature, nn.Parameter):
            self.temperature = temperature
        else:
            self.temperature = nn.Parameter(temperature * torch.ones([]), requires_grad=requires_grad)

        self.logit_scale = torch.log(self.temperature)

    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
    ) -> Tensor:
        self.temperature.data.clamp_(self.temperature_min, self.temperature_max)
        self.logit_scale = torch.log(self.temperature)

        return nt_xent_loss(
            embeddings_a=embeddings_a,
            embeddings_b=embeddings_b,
            temperature=self.temperature,
        )
