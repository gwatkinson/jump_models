import torch
import torch.nn.functional as F
from torch import nn


class VariationalAutoEncoderLoss(nn.Module):
    def __init__(
        self,
        emb_dim,
        similarity,
        beta: float = 1,
        norm: bool = True,
        detach_target: bool = False,
        latent_dim=None,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.similarity = similarity
        self.detach_target = detach_target
        self.beta = beta
        self.norm = norm

        self.latent_dim = latent_dim or emb_dim // 4

        self.criterion = None
        if similarity == "l1":
            self.criterion = nn.L1Loss()
        elif similarity == "l2":
            self.criterion = nn.MSELoss()
        elif similarity == "cosine":
            self.criterion = nn.CosineEmbeddingLoss(dim=-1)

        self.fc_mu = nn.Linear(self.emb_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.emb_dim),
        )

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.norm:
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)

        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y).mean()
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss_dict = {
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "loss": reconstruction_loss + self.beta * kl_loss,
        }

        return loss_dict


class GraphImageVariatonalEncoderLoss(VariationalAutoEncoderLoss):
    def forward(self, image, compound, **kwargs):
        return super().forward(compound, image)


class ImageGraphVariatonalEncoderLoss(VariationalAutoEncoderLoss):
    def forward(self, image, compound, **kwargs):
        return super().forward(image, compound)
