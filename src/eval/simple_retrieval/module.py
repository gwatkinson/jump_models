"""LightningModule for Jump MOA datasets evalulation."""

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.retrieval import (
    RetrievalFallOut,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)


class IDRRetrievalModule(LightningModule):
    retrieval_metrics = MetricCollection(
        {
            "RetrievalMRR": RetrievalMRR(),
            "RetrievalHitRate_top_1": RetrievalHitRate(top_k=1),
            "RetrievalHitRate_top_3": RetrievalHitRate(top_k=3),
            "RetrievalHitRate_top_5": RetrievalHitRate(top_k=5),
            "RetrievalHitRate_top_10": RetrievalHitRate(top_k=10),
            "RetrievalFallOut_top_1": RetrievalFallOut(top_k=1),
            "RetrievalFallOut_top_5": RetrievalFallOut(top_k=5),
            "RetrievalMAP_top_1": RetrievalMAP(top_k=1),
            "RetrievalMAP_top_5": RetrievalMAP(top_k=5),
            "RetrievalPrecision_top_1": RetrievalPrecision(top_k=1),
            "RetrievalPrecision_top_3": RetrievalPrecision(top_k=3),
            "RetrievalPrecision_top_5": RetrievalPrecision(top_k=5),
            "RetrievalPrecision_top_10": RetrievalPrecision(top_k=10),
            "RetrievalRecall_top_1": RetrievalRecall(top_k=1),
            "RetrievalRecall_top_3": RetrievalRecall(top_k=3),
            "RetrievalRecall_top_5": RetrievalRecall(top_k=5),
            "RetrievalRecall_top_10": RetrievalRecall(top_k=10),
            "RetrievalNormalizedDCG": RetrievalNormalizedDCG(),
        }
    )

    def __init__(
        self,
        cross_modal_module: Optional[LightningModule] = None,
        image_encoder: Optional[nn.Module] = None,
        molecule_encoder: Optional[nn.Module] = None,
        distance_metric: Optional[Callable] = None,
        example_input: Optional[Dict[str, Tensor]] = None,
        example_input_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["cross_modal_module", "image_encoder", "molecule_encoder"])

        if not (image_encoder or cross_modal_module):
            raise ValueError("Either image_encoder or cross_modal_module must be provided.")

        if not (molecule_encoder or cross_modal_module):
            raise ValueError("Either molecule_encoder or cross_modal_module must be provided.")

        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            self.image_encoder = cross_modal_module.image_encoder
            self.image_projection_head = cross_modal_module.image_projection_head

        if molecule_encoder is not None:
            self.molecule_encoder = molecule_encoder
        else:
            self.molecule_encoder = cross_modal_module.molecule_encoder
            self.molecule_projection_head = cross_modal_module.molecule_projection_head

        if not self.image_projection_head.out_features == self.molecule_projection_head.out_features:
            raise ValueError(
                f"Image and molecule encoders must have the same output dimension. Got {self.image_projection_head.out_features} and {self.molecule_projection_head.out_features}."
            )

        self.embedding_dim = self.molecule_projection_head.out_features

        if distance_metric is None:
            self.distance_metric = pairwise_cosine_similarity
        else:
            self.distance_metric = distance_metric

        if example_input is not None:
            self.example_input_array = example_input
        elif example_input_path is not None:
            self.example_input_array = torch.load(example_input_path)

    def forward(self, image: torch.Tensor, compound: torch.Tensor, **kwargs):
        output = {
            "image": self.image_projection_head(self.image_encoder(image)),
            "compound": self.molecule_projection_head(self.molecule_encoder(compound)),
        }
        return output

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = {"dataloader_idx": dataloader_idx, "batch_idx": batch_idx}

        if "compound" in batch:
            output["compound"] = self.molecule_encoder(batch["compound"])
            output["activity_flag"] = batch["activity_flag"]
        if "image" in batch:
            output["image"] = self.image_encoder(batch["image"])

        return output

    def retrieval(self, image_embeddings: torch.Tensor, compound_embeddings: torch.Tensor, activities: torch.Tensor):
        dist = self.distance_metric(compound_embeddings, image_embeddings)
        indexes = torch.arange(dist.shape[1]).expand(dist.shape)
        target = activities.expand(dist.T.shape).T

        gene_metrics = self.retrieval_metrics(preds=dist, target=target, indexes=indexes)

        # self.retrieval_metrics.reset()

        return gene_metrics


def calculate_rank(sim_matrix: Tensor, only_average=False) -> Tensor:
    with torch.no_grad():
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device).repeat(sim_matrix.shape[0], 1).t()

        # X to Y ranking
        row_order_x = sim_matrix.argsort(descending=True, dim=-1)
        x_to_y = (row_order_x == labels).nonzero(as_tuple=True)[1]

        # Y to X ranking
        row_order_y = sim_matrix.argsort(descending=True, dim=0).t()
        y_to_x = (row_order_y == labels).nonzero(as_tuple=True)[1]

        batch_size = sim_matrix.shape[0]

        x_to_y_top1 = (x_to_y == 0).float().mean()
        x_to_y_top1_normed = x_to_y_top1 * batch_size
        x_to_y_top5 = (x_to_y < 5).float().mean()
        x_to_y_top10 = (x_to_y < 10).float().mean()
        x_to_y_mean_pos = 1 + x_to_y.float().mean()
        x_to_y_mean_pos_normed = x_to_y_mean_pos / batch_size

        y_to_x_top1 = (y_to_x == 0).float().mean()
        y_to_x_top1_normed = y_to_x_top1 * batch_size
        y_to_x_top5 = (y_to_x < 5).float().mean()
        y_to_x_top10 = (y_to_x < 10).float().mean()
        y_to_x_mean_pos = 1 + y_to_x.float().mean()
        y_to_x_mean_pos_normed = y_to_x_mean_pos / batch_size

        if only_average:
            return {
                "top1": (x_to_y_top1 + y_to_x_top1) / 2,
                "top1_normed": (x_to_y_top1_normed + y_to_x_top1_normed) / 2,
                "top5": (x_to_y_top5 + y_to_x_top5) / 2,
                "top10": (x_to_y_top10 + y_to_x_top10) / 2,
                "mean_pos": (x_to_y_mean_pos + y_to_x_mean_pos) / 2,
                "mean_pos_normed": (x_to_y_mean_pos_normed + y_to_x_mean_pos_normed) / 2,
            }
        else:
            return {
                "top1": (x_to_y_top1 + y_to_x_top1) / 2,
                "top1_normed": (x_to_y_top1_normed + y_to_x_top1_normed) / 2,
                "top5": (x_to_y_top5 + y_to_x_top5) / 2,
                "top10": (x_to_y_top10 + y_to_x_top10) / 2,
                "mean_pos": (x_to_y_mean_pos + y_to_x_mean_pos) / 2,
                "mean_pos_normed": (x_to_y_mean_pos_normed + y_to_x_mean_pos_normed) / 2,
                "x_to_y_top1": x_to_y_top1,
                "x_to_y_top1_normed": x_to_y_top1_normed,
                "x_to_y_top5": x_to_y_top5,
                "x_to_y_top10": x_to_y_top10,
                "x_to_y_mean_pos": x_to_y_mean_pos,
                "x_to_y_mean_pos_normed": x_to_y_mean_pos_normed,
                "y_to_x_top1": y_to_x_top1,
                "y_to_x_top1_normed": y_to_x_top1_normed,
                "y_to_x_top5": y_to_x_top5,
                "y_to_x_top10": y_to_x_top10,
                "y_to_x_mean_pos": y_to_x_mean_pos,
                "y_to_x_mean_pos_normed": y_to_x_mean_pos_normed,
            }
