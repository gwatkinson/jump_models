from typing import Optional

import torch

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class FillNaNs(torch.nn.Module):
    def __init__(self, nan: float = 0.0, posinf: Optional[float] = None, neginf: Optional[float] = None):
        super().__init__()
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        if inpt.isnan().any():
            logger.info(f"Inpt tensor has NaNs. Fill with {self.nan}.")
            return torch.nan_to_num(inpt, nan=self.nan, posinf=self.posinf, neginf=self.neginf)
        else:
            return inpt
