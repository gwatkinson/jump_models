"""LightningModule for Jump MOA datasets evalulation."""

from typing import Any, Dict, Optional

import torch
from lightning import LightningModule
from torch import Tensor

from src.utils import color_log

py_logger = color_log.get_pylogger(__name__)


class SimpleRetrievalModule(LightningModule):
    def __init__(
        self,
        cross_modal_module: LightningModule,
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
            output["compound_emb"] = self.molecule_projection_head(self.molecule_encoder(batch["compound"]))
        if "image" in batch:
            output["image_emb"] = self.image_projection_head(self.image_encoder(batch["image"]))

        return output
