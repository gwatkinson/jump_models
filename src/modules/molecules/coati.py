"""Molecule encoder using the models from
https://github.com/terraytherapeutics/COATI/tree/main."""

from typing import List, Literal, Optional

import torch
import torch.nn as nn

from src.coati.models.io import load_e3gnn_smiles_clip_e2e
from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


COATI_NAME_TO_URL = {
    "tall_closed": "s3://terray-public/models/tall_closed.pkl",
    "grande_closed": "s3://terray-public/models/grande_closed.pkl",
    "grade_closed_fp": "s3://terray-public/models/grade_closed_fp.pkl",
    "barlow_closed_fp": "s3://terray-public/models/barlow_closed_fp.pkl",
    "barlow_closed": "s3://terray-public/models/barlow_closed.pkl",
    "autoreg_only": "s3://terray-public/models/autoreg_only.pkl",
    "barlow_venti": "s3://terray-public/models/barlow_venti.pkl",
    "grande_open": "s3://terray-public/models/grande_open.pkl",
    "selfies_barlow": "s3://terray-public/models/selfies_barlow.pkl",
}
COATI_MODELS = Literal[
    "tall_closed",
    "grande_closed",
    "grade_closed_fp",
    "barlow_closed_fp",
    "barlow_closed",
    "autoreg_only",
    "barlow_venti",
    "grande_open",
    "selfies_barlow",
]


class COATI(nn.Module):
    def __init__(
        self,
        pretrained_name: COATI_MODELS = "grande_closed",
        out_dim: int = 512,
        padding_length: int = 250,
        freeze: bool = False,
        device: Optional[str] = None,
    ):
        super().__init__()

        self.pretrained_name = pretrained_name
        self.out_dim = out_dim
        self.padding_length = padding_length

        self.device = torch.device(device) if device is not None else None

        encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
            doc_url=COATI_NAME_TO_URL[pretrained_name],
            freeze=freeze,
        )

        tokenizer.n_seq = padding_length
        self.tokenizer = tokenizer

        self.backbone = encoder.xformer
        self.backbone

        self.pretrained_dim = encoder.xformer_config.n_embd

        self.projection_head = nn.Sequential(
            nn.Linear(self.pretrained_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.out_dim, self.out_dim),
        )

        logger.info(f"Using pretrained model: {self.pretrained_name}")

    def to(self, device):  # TODO: remove this once we have a proper device management
        module = super().to(device)
        module.device = device
        module.backbone = module.backbone.to(device)
        module.projection_head = module.projection_head.to(device)
        return module

    def tokenize(self, smiles: List[str]) -> torch.Tensor:
        batch_tokens = torch.tensor(
            [
                self.tokenizer.tokenize_text("[SMILES]" + s + "[STOP]", pad=True)
                if s != "*"
                else self.tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
                for s in smiles
            ],
            # device="cpu",
            dtype=torch.int,
        )

        return batch_tokens

    def extract(self, idx):
        return self.backbone.encode(idx, self.tokenizer)

    def forward(self, smiles: List[str], **kwargs):
        tokens = self.tokenize(smiles)

        if self.device:
            tokens = tokens.to(self.device)

        z = self.extract(tokens)
        z = self.projection_head(z)
        return z
