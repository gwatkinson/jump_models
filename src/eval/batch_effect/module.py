"""LightningModule for the Batch Effect evalulation."""

import copy
from typing import Any, Optional

import torch
from lightning import LightningModule

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class TotalBatchEffectModule(LightningModule):
    def __init__(
        self,
        cross_modal_module: Optional[LightningModule] = None,
        image_encoder_attribute_name: str = "image_encoder",
        image_projection_head_attribute_name: str = "image_projection_head",
        example_input: Optional[torch.Tensor] = None,
        example_input_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # encoder
        self.image_encoder = copy.deepcopy(getattr(cross_modal_module, image_encoder_attribute_name))
        self.image_projection_head = copy.deepcopy(getattr(cross_modal_module, image_projection_head_attribute_name))
        self.model_name = self.image_encoder.__class__.__name__

        self.embedding_dim = self.image_encoder.out_dim

        # example input
        if example_input is not None:
            self.example_input_array = example_input
        elif example_input_path is not None:
            self.example_input_array = torch.load(example_input_path)

    def __repr__(self):
        return f"""{self.__class__.__name__}({self.model_name}({self.embedding_dim}))"""

    def forward(self, image):
        return self.image_encoder(image)

    def predict_step(self, batch: Any, batch_idx: int):
        labels = batch["label"]
        sources = batch["source"]
        batches = batch["batch"]
        wells = batch["well"]
        plates = batch["plate"]
        images = batch["image"]

        logits = self.forward(image=images)
        projections = self.image_projection_head(logits)

        return {
            "label": labels,
            "source": sources,
            "batch": batches,
            "well": wells,
            "plate": plates,
            "embedding": logits,
            "projection": projections,
        }

    def training_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
