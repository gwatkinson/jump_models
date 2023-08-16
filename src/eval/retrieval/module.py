"""LightningModule for Jump MOA datasets evalulation."""
# flake8: noqa

from typing import Any, Callable, Dict, List, Optional, Union

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
        image_encoder_attribute_name: str = "image_encoder",
        molecule_encoder_attribute_name: str = "molecule_encoder",
        image_encoder: Optional[nn.Module] = None,
        molecule_encoder: Optional[nn.Module] = None,
        distance_metric: Optional[Callable] = None,
        example_input: Optional[Dict[str, Tensor]] = None,
        example_input_path: Optional[str] = None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["cross_modal_module", "image_encoder", "molecule_encoder"])

        if not (image_encoder or (cross_modal_module and image_encoder_attribute_name)):
            raise ValueError("Either image_encoder or cross_modal_module must be provided.")

        if image_encoder is not None:
            self.image_encoder = image_encoder
            self.model_name = image_encoder.__class__.__name__
        else:
            self.image_encoder = getattr(cross_modal_module, image_encoder_attribute_name)
            self.model_name = self.image_encoder.__class__.__name__

        if not (molecule_encoder or (cross_modal_module and molecule_encoder_attribute_name)):
            raise ValueError("Either molecule_encoder or cross_modal_module must be provided.")

        if molecule_encoder is not None:
            self.molecule_encoder = molecule_encoder
            self.model_name = molecule_encoder.__class__.__name__
        else:
            self.molecule_encoder = getattr(cross_modal_module, molecule_encoder_attribute_name)
            self.model_name = self.molecule_encoder.__class__.__name__

        if not self.image_encoder.out_dim == self.molecule_encoder.out_dim:
            raise ValueError(
                f"Image and molecule encoders must have the same output dimension. Got {self.image_encoder.out_dim} and {self.molecule_encoder.out_dim}."
            )

        self.embedding_dim = self.image_encoder.out_dim

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
            "image": self.image_encoder(image),
            "compound": self.molecule_encoder(compound),
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
        indexes = torch.arange(dist.shape[1])
        gene_metrics = self.retrieval_metrics(
            preds=dist, target=activities.expand(dist.T.shape).T, indexes=indexes.expand(dist.shape)
        )

        self.retrieval_metrics.reset()

        return gene_metrics
