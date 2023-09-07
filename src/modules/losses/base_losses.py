from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


def uniformity_loss_fn(x1: Tensor, x2: Tensor, t=2) -> Tensor:
    sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
    uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
    sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
    uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
    return (uniformity_x1 + uniformity_x2) / 2


def cov_loss_fn(x):
    batch_size, metric_dim = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (batch_size - 1)
    off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
    return off_diag_cov.pow_(2).sum() / metric_dim


def std_loss_fn(x):
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    return torch.mean(torch.relu(1 - std))


class ClampedParameter:
    def __init__(
        self, value: float, min_value: float, max_value: float, requires_grad: bool = False, name: Optional[str] = None
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.requires_grad = requires_grad
        self.name = name

        self._value = nn.Parameter(value * torch.ones([]), requires_grad=requires_grad)

    @property
    def value(self):
        self.clamp()
        return self._value

    @value.setter
    def value(self, value):
        self._value = nn.Parameter(value * torch.ones([]), requires_grad=self.requires_grad)
        self.clamp()

    def clamp(self):
        if self.requires_grad:
            self._value.data.clamp_(self.min_value, self.max_value)

    def __repr__(self):
        name = f"<ClampedParameter> {self.name}" if self.name else "<ClampedParameter>"
        return f"{name}: {self.value.item()}"


class LossWithTemperature(_Loss):
    def __init__(
        self,
        temperature: float = 0.5,
        temperature_requires_grad: bool = False,
        temperature_min: Optional[float] = 0.0,
        temperature_max: Optional[float] = 100.0,
        **kwargs,
    ):
        super().__init__()
        self.temperature = ClampedParameter(
            value=temperature,
            min_value=temperature_min,
            max_value=temperature_max,
            requires_grad=temperature_requires_grad,
        )

    def forward(z1, z2, **kwargs):
        raise NotImplementedError


class RegularizationLoss(_Loss):
    def __init__(
        self,
        mse_reg: float = 1,
        l1_reg: float = 0,
        uniformity_reg: float = 0,
        variance_reg: float = 0,
        covariance_reg: float = 0,
        name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.mse_reg = mse_reg
        self.l1_reg = l1_reg
        self.name = name or "Regularization"

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        loss = 0
        loss_dict = {}
        if self.mse_reg > 0:
            mse_reg = self.mse_loss(z1, z2)
            loss += self.mse_reg * mse_reg
            loss_dict["mse_loss"] = mse_reg
        if self.l1_reg > 0:
            l1_loss = self.l1_loss(z1, z2)
            loss += self.l1_reg * l1_loss
            loss_dict["l1_loss"] = l1_loss
        if self.variance_reg > 0:
            std_loss = std_loss_fn(z1) + std_loss_fn(z2)
            loss += self.variance_reg * std_loss
            loss_dict["std_loss"] = std_loss
        if self.covariance_reg > 0:
            cov_loss = cov_loss_fn(z1) + cov_loss_fn(z2)
            loss += self.covariance_reg * cov_loss
            loss_dict["cov_loss"] = cov_loss
        if self.uniformity_reg > 0:
            uniformity_loss = uniformity_loss_fn(z1, z2)
            loss += self.uniformity_reg * uniformity_loss
            loss_dict["uniformity_loss"] = uniformity_loss

        loss_dict["loss"] = loss

        return loss_dict


class CombinationLoss(_Loss):
    def __init__(
        self,
        losses: Dict[_Loss],
        norm: bool = True,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.norm = norm
        self.losses = losses
        self.weights = weights or [1 / len(losses)] * len(losses)

        if self.norm:
            for loss_fn in self.losses.values():
                if hasattr(loss_fn, "norm"):
                    loss_fn.norm = False  # To avoid double normalization

    def forward(self, z1, z2, **kwargs) -> Tensor:
        if self.norm:
            z1 = F.normalize(z1, dim=-1, p=2)
            z2 = F.normalize(z2, dim=-1, p=2)

        loss_dict = {}
        keys = []
        loss = 0
        for loss_name, weight in zip(self.losses, self.weights):
            if hasattr(self.losses[loss_name], "name"):
                key = self.losses[loss_name].name
            else:
                key = loss_name

            if key in keys:
                key += "_"  # Add underscore to avoid overwriting
            keys.append(key)

            loss_results = self.losses[loss_name](z1, z2, **kwargs)

            if isinstance(loss_results, dict):
                loss_value = loss_results["loss"]
                log_dict = {f"{key}/{k}": v for k, v in loss_results.items()}
            else:
                loss_value = loss_results
                log_dict = {f"{key}/loss": loss_value}

            loss_dict.update(log_dict)

            loss += weight * loss_value

        loss_dict["loss"] = loss

        return loss_dict


class RegLoss(CombinationLoss):
    def __init__(
        self,
        loss_fn: _Loss,
        alpha: float = 0.05,
        norm: bool = True,
        mse_reg: float = 1.0,
        l1_reg: float = 0,
        uniformity_reg: float = 0,
        variance_reg: float = 1.0,
        covariance_reg: float = 0.05,
        **kwargs,
    ):
        weights = [1 - alpha, alpha]

        losses = {
            "temp_loss": loss_fn,
            "reg_loss": RegularizationLoss(
                mse_reg=mse_reg,
                l1_loss=l1_reg,
                uniformity_reg=uniformity_reg,
                variance_reg=variance_reg,
                covariance_reg=covariance_reg,
            ),
        }

        super().__init__(losses=losses, weights=weights, norm=norm)


class RegWithTemperatureLoss(RegLoss):
    loss_fn = None
    name = None

    def __init__(
        self,
        alpha: float = 0.05,
        norm: bool = True,
        temperature: float = 0.5,
        mse_reg: float = 1,
        l1_reg: float = 0,
        uniformity_reg: float = 0,
        variance_reg: float = 1.0,
        covariance_reg: float = 0.05,
        temperature_requires_grad: bool = False,
        temperature_min: Optional[float] = 0.0,
        temperature_max: Optional[float] = 100.0,
        **kwargs,
    ):
        loss_fn = self.loss_fn(
            norm=False,
            temperature=temperature,
            temperature_requires_grad=temperature_requires_grad,
            temperature_min=temperature_min,
            temperature_max=temperature_max,
            name=self.name,
            **kwargs,
        )

        self.temperature = loss_fn.temperature

        super().__init__(
            loss_fn=loss_fn,
            alpha=alpha,
            norm=norm,
            mse_reg=mse_reg,
            l1_reg=l1_reg,
            uniformity_reg=uniformity_reg,
            variance_reg=variance_reg,
            covariance_reg=covariance_reg,
        )
