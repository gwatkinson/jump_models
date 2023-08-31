from typing import Literal

import torch

from src.coati.models.io import COATI_MODELS, load_coati_tokenizer, load_model_doc
from src.modules.compound_transforms.base_compound_transform import DefaultCompoundTransform


class COATITransform(DefaultCompoundTransform):
    """Pass the string through the COATI model."""

    def __init__(
        self,
        pretrained_name: COATI_MODELS = "grande_closed",
        padding_length: int = 250,
        compound_str_type: Literal["inchi", "smiles", "selfies", "smarts"] = "smiles",
        model_dir: str = "./models",
    ):
        super().__init__(compound_str_type)

        self.pretrained_name = pretrained_name

        if self.pretrained_name not in COATI_MODELS:
            raise ValueError(f"pretrained_name must be one of {COATI_MODELS}")

        model_doc = load_model_doc(pretrained_name, model_dir=model_dir)
        tokenizer = load_coati_tokenizer(model_doc)

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
