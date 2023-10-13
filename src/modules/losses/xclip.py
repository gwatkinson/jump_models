# contrastive loss inspired by https://github.com/lucidrains/x-clip/blob/main/x_clip/x_clip.py#L412

import copy
import math
from contextlib import contextmanager
from functools import partial, wraps

import torch
import torch.distributed as distributed
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum, nn
from torch.utils.checkpoint import checkpoint


class CLIP(nn.Module):
    def __init__(
        self,
        *,
        image_encoder=None,
        molecule_encoder=None,
        dim_image=512,
        dim_molecule=512,
        dim_latent=512,
        molecule_has_cls_token=False,
        visual_has_cls_token=False,
        use_all_token_embeds=False,
        temperature=0.1,
        decoupled_contrastive_learning=False,
        extra_latent_projection=False,
        use_mgm=False,
        use_mim=False,
        multimodal_ssl=True,
        image_ssl_loss_weight=0.05,
        molecule_ssl_loss_weight=0.05,
        multiview_loss_weight=0.1,
        sim_reg_loss_weight=0.0,
        checkpoint_during_training=False,
        **kwargs,
    ):
        super().__init__()

        assert use_all_token_embeds or (
            molecule_has_cls_token and visual_has_cls_token
        ), "CLS token must be included on both vision and molecule encoders if you are not using fine-grained contrastive learning loss"
