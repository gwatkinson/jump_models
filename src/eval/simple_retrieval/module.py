"""LightningModule for Jump MOA datasets evalulation."""

from typing import Any, Callable, Dict, Optional

import torch
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
        cross_modal_module: LightningModule,
        distance_metric: Optional[Callable] = None,
        example_input: Optional[Dict[str, Tensor]] = None,
        example_input_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["cross_modal_module"])

        self.image_encoder = cross_modal_module.image_encoder
        self.image_projection_head = cross_modal_module.image_projection_head

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
        output = {
            "dataloader_idx": dataloader_idx,
            "batch_idx": batch_idx,
            "compound_str": batch["compound_str"],
            "image_id": batch["image_id"],
        }

        if "compound" in batch:
            output["compound"] = self.molecule_projection_head(self.molecule_encoder(batch["compound"]))
        if "image" in batch:
            output["image"] = self.image_projection_head(self.image_encoder(batch["image"]))

        return output

    def retrieval(self, image_embeddings: torch.Tensor, compound_embeddings: torch.Tensor, activities: torch.Tensor):
        dist = self.distance_metric(compound_embeddings, image_embeddings)
        indexes = torch.arange(dist.shape[1]).expand(dist.shape)
        target = activities.expand(dist.T.shape).T

        gene_metrics = self.retrieval_metrics(preds=dist, target=target, indexes=indexes)

        # self.retrieval_metrics.reset()

        return gene_metrics
