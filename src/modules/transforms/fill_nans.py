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

    def forward(self, inpt: torch.Tensor, input_id: Optional[str]) -> torch.Tensor:
        if inpt.isnan().any():
            mid_str = f" of {input_id}" if input_id is not None else ""
            logger.info(f"Inpt tensor{mid_str} has NaNs. Fill with {self.nan}.")
            return torch.nan_to_num(inpt, nan=self.nan, posinf=self.posinf, neginf=self.neginf)
        else:
            return inpt
