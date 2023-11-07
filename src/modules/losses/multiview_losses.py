# from torch.distributions import MultivariateNormal
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from src.modules.losses.base_losses import LossWithTemperature, cov_loss_fn, std_loss_fn, uniformity_loss_fn
from src.modules.losses.contrastive_losses import calculate_rank


class MultiviewRegLoss(_Loss):
    def __init__(
        self,
        mse_reg: float = 0.5,
        l1_reg: float = 0,
        uniformity_reg: float = 0,
        variance_reg: float = 1,
        covariance_reg: float = 0.25,
        conformer_variance_reg: float = 1,
        name: str = "MultiviewRegularization",
        **kwargs,
    ):
        super().__init__()
        self.mse_reg = mse_reg
        self.l1_reg = l1_reg
        self.uniformity_reg = uniformity_reg
        self.variance_reg = variance_reg
        self.covariance_reg = covariance_reg
        self.conformer_variance_reg = conformer_variance_reg

        self.name = name

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, z1, z2, **kwargs) -> Tensor:
        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        loss = 0
        loss_dict = {}

        if self.mse_reg > 0:
            mse_loss = self.mse_reg * self.mse_loss(z1, z2)
            loss += mse_loss
            loss_dict["mse_loss"] = mse_loss
        if self.l1_reg > 0:
            l1_loss = self.l1_reg * self.l1_loss(z1, z2)
            loss += l1_loss
            loss_dict["l1_loss"] = l1_loss
        if self.variance_reg > 0:
            std_loss = self.variance_reg * (std_loss_fn(z1) + std_loss_fn(z2))
            loss += std_loss
            loss_dict["std_loss"] = std_loss
        if self.conformer_variance_reg > 0:
            std = torch.sqrt(z2.var(dim=1) + 1e-04)
            std_conf_loss = self.conformer_variance_reg * torch.mean(torch.relu(1 - std))
            loss += std_conf_loss
            loss_dict["std_conf_loss"] = std_conf_loss
        if self.covariance_reg > 0:
            cov_loss = self.covariance_reg * (cov_loss_fn(z1) + cov_loss_fn(z2))
            loss += cov_loss
            loss_dict["cov_loss"] = cov_loss
        if self.uniformity_reg > 0:
            uniformity_loss = self.uniformity_reg * uniformity_loss_fn(z1, z2)
            loss += uniformity_loss
            loss_dict["uniformity_loss"] = uniformity_loss

        loss_dict["loss"] = loss

        return loss_dict


class NTXentMultiplePositives(LossWithTemperature):
    def __init__(
        self,
        norm: bool = True,
        temperature: float = 0.5,
        return_rank: bool = False,
        eps: float = 1e-8,
        name: str = "NTXentMultiplePositives",
        **kwargs,
    ) -> None:
        super().__init__(temperature=temperature, **kwargs)

        self.norm = norm
        self.eps = eps
        self.name = name
        self.return_rank = return_rank  # TODO for multiview

    def forward(self, image_emb, compound_emb, **kwargs) -> Tensor:
        """
        :param z1: batchsize, metric dim
        :param z2: batchsize, num_conformers, metric dim

        always the same number of conformers are required
        """
        z1, z2 = compound_emb, image_emb

        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        _, num_conformers, _ = z2.size()

        sim_matrix = torch.einsum("ik,juk->iju", z1, z2)  # [batch_size, batch_size, num_conformers]

        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum("i,ju->iju", z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.temperature)  # [batch_size, batch_size, num_conformers]

        pos_sim = sim_matrix[range(batch_size), range(batch_size), :]  # [batch_size, num_conformers]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  # [batch_size, num_conformers]
        loss = -torch.log(loss).mean()

        loss_dict = {"loss": loss}

        if self.return_rank:
            results_dict = defaultdict(list)
            for i in range(num_conformers):
                sub_sim = sim_matrix[:, :, i]
                results = calculate_rank(sub_sim, only_average=True)
                for k, v in results.items():
                    results_dict[k].append(v)
                    loss_dict[f"{k}_{i}"] = v

            for k, v in results_dict.items():
                loss_dict[f"{k}_mean"] = torch.stack(v).mean()
                loss_dict[f"{k}_std"] = torch.stack(v).std()

        return loss_dict


class MultiviewIntraModalNTXentLoss(LossWithTemperature):
    def __init__(
        self,
        norm: bool = True,
        temperature: float = 0.5,
        return_rank: bool = False,
        lambda_image: float = 0.5,
        eps: float = 1e-8,
        name: str = "MultiviewIntraModalNTXentLoss",
        **kwargs,
    ):
        super().__init__(temperature=temperature, **kwargs)

        self.norm = norm
        self.eps = eps
        self.name = name
        self.lambda_image = lambda_image  # factor for intra image loss
        self.return_rank = return_rank

    def forward(self, image_emb, compound_emb, **kwargs):
        """
        :param z1: batchsize, metric dim
        :param z2: batchsize, num_conformers, metric dim
        """
        z1, z2 = compound_emb, image_emb

        batch_size, metric_dim = z1.size()
        z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        batch_size, num_conformers, metric_dim = z2.size()

        z1 = z1.unsqueeze(1)  # [batch_size, 1, metric_dim]
        z3 = torch.cat([z1, z2], dim=1)  # [batch_size, num_conformers+1, metric_dim]

        if self.norm:
            z3 = F.normalize(z3, dim=2)

        possible_pairs = torch.triu_indices(num_conformers + 1, num_conformers + 1, 1)  # [2, num_pairs]
        num_pairs = possible_pairs.size(1)

        loss_value = 0
        loss_dict = {}
        results_dict = defaultdict(list)
        for _ in range(num_pairs):
            i, j = possible_pairs[:, _]
            z1p = z3[:, i, :].squeeze()
            z2p = z3[:, j, :].squeeze()

            sim_matrix = torch.einsum("ik,jk->ij", z1p, z2p)  # [batch_size, batch_size]
            sim_matrix = torch.exp(sim_matrix / self.temperature)  # [batch_size, batch_size]

            pos_sim = sim_matrix[range(batch_size), range(batch_size)]  # [batch_size]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)  # [batch_size]
            loss = -torch.log(loss).mean()

            if i != 0 and j != 0:
                loss = self.lambda_image * loss

            loss_dict[f"loss_{i}_{j}"] = loss
            loss_value += loss

            if self.return_rank:
                results = calculate_rank(sim_matrix, only_average=True)
                for k, v in results.items():
                    results_dict[k].append(v)
                    loss_dict[f"{k}_{i}_{j}"] = v

        loss_dict["loss"] = loss_value

        if self.return_rank:
            for k, v in results_dict.items():
                loss_dict[f"{k}_mean"] = torch.stack(v).mean()
                loss_dict[f"{k}_std"] = torch.stack(v).std()

        return loss_dict


class KLDivergenceMultiplePositives(LossWithTemperature):
    def __init__(
        self,
        emb_dim: int,
        norm: bool = True,
        temperature: float = 0.5,
        return_rank: bool = False,
        eps: float = 1e-8,
        name: str = "KLDivergenceMultiplePositives",
        **kwargs,
    ) -> None:
        super().__init__(temperature=temperature, **kwargs)

        self.norm = norm
        self.eps = eps
        self.name = name
        self.return_rank = return_rank

        self.mu_fc = nn.Linear(emb_dim, emb_dim)
        self.sigma_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, image_emb, compound_emb, **kwargs) -> Tensor:
        """
        :param z1: batchsize, 2, metric dim
        :param z2: batchsize, num_conformers, metric dim
        """
        z1 = torch.stack([self.mu_fc(compound_emb), self.sigma_fc(compound_emb)], dim=1)
        z2 = image_emb

        batch_size, num_conformers, metric_dim = z2.size()

        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_stds = torch.exp(z1[:, 1, :])  # [batch_size, metric_dim]

        # z2 = z2.view(batch_size, -1, metric_dim)  # [batch_size, num_conformers, metric_dim]
        z2_means = z2.mean(1)  # [batch_size, metric_dim]
        z2_stds = z2.std(1)  # [batch_size, metric_dim]

        kl_div_kernel = []
        for i, z1_mean in enumerate(z1_means):  # batch_size
            for j, z2_mean in enumerate(z2_means):  # batch_size
                z1_std = z1_stds[i]  # [metric_dim]
                z2_std = z2_stds[j] + 1e-5  # [metric_dim]
                p = torch.distributions.MultivariateNormal(z1_mean, torch.diag_embed(z1_std))
                q = torch.distributions.MultivariateNormal(z2_mean, torch.diag_embed(z2_std))
                kl_divergence = torch.distributions.kl.kl_divergence(p, q)
                kl_div_kernel.append(kl_divergence)

        kl_div_kernel = torch.stack(kl_div_kernel)  # [batch_size*batch_size]
        kl_div_kernel = kl_div_kernel.view(batch_size, batch_size)

        sim_matrix = torch.nan_to_num(torch.exp(kl_div_kernel / self.temperature))  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        return loss


class NTXentLikelihoodLoss(LossWithTemperature):
    def __init__(
        self,
        emb_dim: int,
        norm: bool = True,
        temperature: float = 0.5,
        return_rank: bool = False,
        eps: float = 1e-8,
        name: str = "NTXentLikelihood",
        **kwargs,
    ) -> None:
        super().__init__(temperature=temperature, **kwargs)

        self.norm = norm
        self.eps = eps
        self.name = name
        self.return_rank = return_rank

        self.mu_fc = nn.Linear(emb_dim, emb_dim)
        self.sigma_fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, image_emb, compound_emb, **kwargs) -> Tensor:
        """
        :param z1: batchsize, metric dim
        :param z2: batchsize, num_conformers, metric dim
        """
        z1 = torch.stack([self.mu_fc(compound_emb), self.sigma_fc(compound_emb)], dim=1)
        z2 = image_emb

        batch_size, num_conformers, metric_dim = z2.size()

        z1_means = z1[:, 0, :]  # [batch_size, metric_dim]
        z1_stds = torch.exp(z1[:, 1, :])  # [batch_size, metric_dim]

        likelihood_kernel = []
        for i, z1_mean in enumerate(z1_means):
            z1_std = z1_stds[i]  # [metric_dim]
            p = torch.distributions.Normal(z1_mean, z1_std)
            for _, z2_elem in enumerate(z2):
                prob = torch.exp(p.log_prob(z2_elem))
                likelihood_kernel.append(prob.mean())
        likelihood_kernel = torch.stack(likelihood_kernel)
        likelihood_kernel = likelihood_kernel.view(batch_size, batch_size)

        sim_matrix = torch.exp(likelihood_kernel / self.temperature)  # [batch_size, batch_size]
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()

        return {"loss": loss}
