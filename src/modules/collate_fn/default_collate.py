from typing import Callable, Dict, Optional, Tuple, Type, Union

from dgl import DGLGraph
from dgl import batch as dgl_batch
from torch.utils.data._utils.collate import collate
from torch.utils.data._utils.collate import default_collate_fn_map as _default_collate_fn_map


def collate_dgl_graph_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return dgl_batch(batch)


_default_collate_fn_map[
    DGLGraph
] = collate_dgl_graph_fn  # Add DGLGraph to default_collate_fn_map of torch.utils.data._utils.collate


def default_collate(x):
    return collate(x, collate_fn_map=_default_collate_fn_map)
