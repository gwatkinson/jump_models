"""Molecule encoder using the models from
https://github.com/terraytherapeutics/COATI/tree/main."""

import torch.nn as nn

from src.coati.models.io import COATI_MODELS, load_coati_model, load_coati_tokenizer, load_model_doc
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class COATI(nn.Module):
    def __init__(
        self,
        pretrained_name: COATI_MODELS = "grande_closed",
        out_dim: int = 512,
        padding_length: int = 250,
        model_dir: str = "./models",
    ):
        super().__init__()

        self.pretrained_name = pretrained_name
        if self.pretrained_name not in COATI_MODELS:
            raise ValueError(f"pretrained_name must be one of {COATI_MODELS}")

        model_doc = load_model_doc(pretrained_name, model_dir=model_dir)
        tokenizer = load_coati_tokenizer(model_doc)
        encoder = load_coati_model(model_doc, model_type="default")

        self.padding_length = padding_length
        tokenizer.n_seq = padding_length
        self.tokenizer = tokenizer

        self.backbone = encoder.xformer

        self.out_dim = out_dim
        self.pretrained_dim = encoder.xformer_config.n_embd

        self.projection_head = nn.Sequential(
            nn.Linear(self.pretrained_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.out_dim, self.out_dim),
        )

        logger.info(f"Using pretrained model: {self.pretrained_name}")

    def extract(self, idx):
        return self.backbone.encode(idx, self.tokenizer)

    def forward(self, tokens, **kwargs):
        z = self.extract(tokens)
        z = self.projection_head(z)
        return z
