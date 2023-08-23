# Heavily inspired from https://github.com/HannesStark/3DInfomax/blob/master/commons/losses.py

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss


def uniformity_loss(x1: Tensor, x2: Tensor, t=2) -> Tensor:
    sq_pdist_x1 = torch.pdist(x1, p=2).pow(2)
    uniformity_x1 = sq_pdist_x1.mul(-t).exp().mean().log()
    sq_pdist_x2 = torch.pdist(x2, p=2).pow(2)
    uniformity_x2 = sq_pdist_x2.mul(-t).exp().mean().log()
    return (uniformity_x1 + uniformity_x2) / 2


def cov_loss(x):
    batch_size, metric_dim = x.size()
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (batch_size - 1)
    off_diag_cov = cov.flatten()[:-1].view(metric_dim - 1, metric_dim + 1)[:, 1:].flatten()
    return off_diag_cov.pow_(2).sum() / metric_dim


def std_loss(x):
    std = torch.sqrt(x.var(dim=0) + 1e-04)
    return torch.mean(torch.relu(1 - std))


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


class CosineSimilarityLoss(_Loss):
    def __init__(self, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super().__init__()
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        # see the "Bootstrap your own latent" paper equation 2 for the loss"
        # this loss is equivalent to 2 - 2*cosine_similarity
        x = F.normalize(z1, dim=-1, p=2)
        y = F.normalize(z2, dim=-1, p=2)
        loss = (((x - y) ** 2).sum(dim=-1)).mean()
        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class InfoNCE(_Loss):
    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super().__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / torch.einsum("i,j->ij", z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1))
        loss = -torch.log(loss).mean()
        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXent(_Loss):
    def __init__(self, norm: bool = True, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0) -> None:
        super().__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, _ = z1.size()
        sim_matrix = torch.einsum("ik,jk->ij", z1, z2)

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=1)
            sim_matrix = sim_matrix / (torch.einsum("i,j->ij", z1_abs, z2_abs) + 1e-8)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class NTXentMultiplePositives(_Loss):
    def __init__(
        self,
        norm: bool = True,
        tau: float = 0.5,
        uniformity_reg=0,
        variance_reg=0,
        covariance_reg=0,
        conformer_variance_reg=0,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.conformer_variance_reg = conformer_variance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        """
        :param z1: batchsize, metric dim
        :param z2: batchsize*num_conformers, metric dim
        """
        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]

        sim_matrix = torch.einsum("ik,juk->iju", z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum("i,ju->iju", z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)  # [batch_size, batch_size, num_conformers]

        sim_matrix = sim_matrix.sum(dim=2)  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)  # [batch_size]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.conformer_variance_reg > 0:
            std = torch.sqrt(z2.var(dim=1) + 1e-04)
            std_conf_loss = torch.mean(torch.relu(1 - std))
            loss += self.conformer_variance_reg * std_conf_loss
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss


class KLDivergenceMultiplePositives(_Loss):
    """
    Normalized Temperature-scaled Cross Entropy Loss from SimCLR paper
    Args:
        z1, z2: Tensor of shape [batch_size, z_dim]
        tau: Float. Usually in (0,1].
        norm: Boolean. Whether to apply normlization.
    """

    def __init__(
        self, norm: bool = False, tau: float = 0.5, uniformity_reg=0, variance_reg=0, covariance_reg=0
    ) -> None:
        super().__init__()
        self.norm = norm
        self.tau = tau
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg

    def forward(self, z1, z2, **kwargs) -> Tensor:
        """
        :param z1: batchsize, metric dim*2
        :param z2: batchsize*num_conformers, metric dim
        """
        batch_size, _ = z1.size()
        _, metric_dim = z2.size()

        z1 = z1.view(batch_size, 2, metric_dim)
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        if self.norm:
            z1 = F.normalize(z1, dim=2)
            z2 = F.normalize(z2, dim=2)
        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_vars = torch.exp(z1[:, 1, :])  # [batch_size, metric_dim]
        z2_means = z2.mean(1)  # [batch_size, metric_dim]
        z2_vars = z2.var(1) + 1e-6  # [batch_size, metric_dim]
        try:
            normal1 = MultivariateNormal(z1_means, torch.diag_embed(z1_vars))
        except Exception:
            print(z1_vars)
        try:
            normal2 = MultivariateNormal(z2_means, torch.diag_embed(z2_vars))
        except Exception:
            print(z2_vars)
        # kl_div = torch.distributions.kl_divergence(normal1, normal2)
        kl_div = torch.distributions.kl_divergence(normal2, normal1)
        loss = kl_div.mean()

        if self.variance_reg > 0:
            loss += self.variance_reg * (std_loss(z1) + std_loss(z2))
        if self.covariance_reg > 0:
            loss += self.covariance_reg * (cov_loss(z1) + cov_loss(z2))
        if self.uniformity_reg > 0:
            loss += self.uniformity_reg * uniformity_loss(z1, z2)
        return loss
