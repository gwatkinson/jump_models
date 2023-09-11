# Inspired from https://github.com/prokia/MIGA/blob/main/pretrain.py#L54

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.functional import pairwise_cosine_similarity

from src.modules.layers.fusion import DeepsetFusionWithTransformer


class DeepSetFusion(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        latent_dim: Optional[int] = None,
        num_transformer_att_heads: int = 8,
        num_transformer_layers: int = 1,
        apply_attention: bool = False,
        attention_dim: Optional[int] = None,
        modality_normalize: bool = False,
        norm_factor: float = 2.0,
    ):
        super().__init__()

        latent_dim = latent_dim or emb_dim // 4

        self.image_proj = nn.Linear(emb_dim, latent_dim)
        self.graph_proj = nn.Linear(emb_dim, latent_dim)

        mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 2 * latent_dim),
        )

        transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=latent_dim, nhead=num_transformer_att_heads, batch_first=True
            ),
            num_layers=num_transformer_layers,
            norm=nn.LayerNorm(latent_dim),
        )
        self.fusion = DeepsetFusionWithTransformer(
            channel_to_encoder_dim={
                "image": latent_dim,
                "graph": latent_dim,
            },
            mlp=mlp,
            pooling_function=transformer,
            apply_attention=apply_attention,
            attention_dim=attention_dim,
            modality_normalize=modality_normalize,
            norm_factor=norm_factor,
            use_auto_mapping=False,
        )

        self.out_dim = 2 * latent_dim

    def forward(self, graph_emb, img_emb):
        graph_emb = self.graph_proj(graph_emb)
        img_emb = self.image_proj(img_emb)
        ins = {
            "image": img_emb,
            "graph": graph_emb,
        }

        return self.fusion(ins)


class CatFusion:
    def __init__(self, embedding_dim: int):
        self.out_dim = 2 * embedding_dim

    def __call__(self, graph_emb, img_emb):
        return torch.cat([graph_emb, img_emb], dim=1)


class GraphImageMatchingLoss(_Loss):
    def __init__(
        self,
        embedding_dim: int,
        norm: bool = True,
        name: str = "GraphImageMatchingLoss",
        fusion_layer=None,
        **kwargs,
    ):
        super().__init__()
        self.norm = norm
        self.name = name

        if fusion_layer in [None, "concat", "cat"]:
            self.fusion_layer = CatFusion(embedding_dim)
        elif fusion_layer == "deepset":
            self.fusion_layer = DeepSetFusion(embedding_dim)
        else:
            self.fusion_layer = fusion_layer

        self.feature_dim = self.fusion_layer.out_dim

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 2),
        )

        self.auroc = BinaryAUROC()
        self.accuracy = BinaryAccuracy()
        self.recall = BinaryRecall()
        self.precision = BinaryPrecision()
        self.f1_score = BinaryF1Score()

    def sample_negatives(self, x_emb, y_emb):
        batch_size = x_emb.size(0)

        with torch.no_grad():
            weights_x2y = F.softmax(pairwise_cosine_similarity(x_emb, y_emb), dim=-1)
            weights_x2y.fill_diagonal_(0)

        neg_y = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_x2y[b], 1).item()
            neg_y.append(y_emb[neg_idx])
        neg_y = torch.stack(neg_y, dim=0)

        return neg_y

    def stack_embeddings(self, graph_emb, img_emb):
        batch_size = graph_emb.size(0)

        neg_images = self.sample_negatives(graph_emb, img_emb)  # [batch_size, metric_dim]
        neg_graphs = self.sample_negatives(img_emb, graph_emb)

        graphs = torch.cat([graph_emb, neg_graphs, graph_emb], dim=0)
        images = torch.cat([img_emb, img_emb, neg_images], dim=0)

        labels = torch.cat(
            [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)], dim=0
        ).to(graph_emb.device)

        features = self.fusion_layer(graphs, images)

        return features, labels

    def forward(self, graph_emb, img_emb, **kwargs) -> Tensor:
        batch_size, metric_dim = graph_emb.size()

        if self.norm:
            graph_emb = F.normalize(graph_emb, dim=-1, p=2)
            img_emb = F.normalize(img_emb, dim=-1, p=2)

        # Sample negatives and stack
        gim_features, gim_labels = self.stack_embeddings(graph_emb, img_emb)  # [3*batch_size, out_dim]

        gim_preds = self.head(gim_features)  # [3*batch_size, 2]

        gim_cross_entropy = F.cross_entropy(gim_preds, gim_labels)

        auroc = self.auroc(gim_preds[:, 1], gim_labels)
        accuracy = self.accuracy(gim_preds[:, 1], gim_labels)
        recall = self.recall(gim_preds[:, 1], gim_labels)
        precision = self.precision(gim_preds[:, 1], gim_labels)
        f1_score = self.f1_score(gim_preds[:, 1], gim_labels)

        loss_dict = {
            "loss": gim_cross_entropy,
            "auroc": auroc,
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
        }

        return loss_dict
