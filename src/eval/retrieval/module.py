"""LightningModule for Jump MOA datasets evalulation."""

import contextlib
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor


class IDRRetrievalModule(LightningModule):
    def __init__(
        self,
        cross_modal_module: Optional[LightningModule] = None,
        image_encoder: Optional[nn.Module] = None,
        molecule_encoder: Optional[nn.Module] = None,
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

        self.embedding_dim = self.image_projection_head.out_features

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
            compound = batch["compound"]
            with contextlib.suppress(Exception):
                compound = compound.squeeze()

            output["compound"] = self.molecule_projection_head(self.molecule_encoder(compound))
            output["activity_flag"] = batch["activity_flag"].squeeze()

        if "image" in batch:
            output["image"] = self.image_projection_head(self.image_encoder(batch["image"]))

        return output

    # def retrieval(self, image_embeddings: torch.Tensor, compound_embeddings: torch.Tensor, activities: torch.Tensor):
    #     dist = self.distance_metric(compound_embeddings, image_embeddings)  # I x C
    #     indexes = torch.arange(dist.shape[1]).expand(dist.shape)  # I x C
    #     target = activities.expand(dist.T.shape).T  # I x C

    #     gene_metrics = self.retrieval_metrics(preds=dist, target=target, indexes=indexes)

    #     # self.retrieval_metrics.reset()

    #     return gene_metrics
