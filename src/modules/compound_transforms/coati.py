from typing import Literal

import torch

from src.coati.models.io import load_e3gnn_smiles_clip_e2e
from src.modules.compound_transforms.base_compound_transform import DefaultCompoundTransform
from src.modules.molecules.coati import COATI_MODELS, COATI_NAME_TO_URL


class COATITransform(DefaultCompoundTransform):
    """Pass the string through the COATI model."""

    def __init__(
        self,
        pretrained_name: COATI_MODELS = "grande_closed",
        padding_length: int = 250,
        compound_str_type: Literal["inchi", "smiles", "selfies", "smarts"] = "smiles",
    ):
        super().__init__(compound_str_type)

        self.pretrained_name = pretrained_name

        encoder, tokenizer = load_e3gnn_smiles_clip_e2e(
            doc_url=COATI_NAME_TO_URL[pretrained_name], freeze=False, device="cpu"
        )

        del encoder

        self.padding_length = padding_length
        tokenizer.n_seq = padding_length
        self.tokenizer = tokenizer

    def mol_to_feat(self, smiles: str) -> torch.Tensor:
        tokens = torch.tensor(
            [
                self.tokenizer.tokenize_text("[SMILES]" + smiles + "[STOP]", pad=True)
                if smiles != "*"
                else self.tokenizer.tokenize_text("[SMILES]C[STOP]", pad=True)
            ],
            dtype=torch.int,
        ).squeeze()

        return tokens
