# Taken from https://github.com/facebookresearch/multimodal/blob/main/torchmultimodal/modules/fusions/concat_fusion.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# (https://github.com/facebookresearch/multimodal/blob/main/LICENSE)

from typing import Dict

import torch
from torch import nn


class ConcatFusionModule(nn.Module):
    """Module to fuse modalities via concatenation. Sorted by keys for
    consistency.

    Inputs:
        embeddings (Dict[str, Tensor]): A dictionary mapping modalities to their
            tensor representations.
    """

    def __init__(self, projection: nn.Module = None):
        super().__init__()
        self.projection = projection or nn.Identity()

    def forward(self, embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        concatenated_in = torch.cat([embeddings[k] for k in sorted(embeddings.keys())], dim=-1)
        return self.projection(concatenated_in)