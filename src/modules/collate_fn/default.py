from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from dgl import DGLGraph
from dgl import batch as dgl_batch
from torch.utils.data._utils.collate import collate
from torch.utils.data._utils.collate import default_collate_fn_map as _default_collate_fn_map


@dataclass
class SmilesList:
    smiles_list: List[str]


def collate_dgl_graph_fn(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return dgl_batch(batch)


def collate_smiles_list(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return [i.smiles_list for i in batch]


# Add DGLGraph to default_collate_fn_map of torch.utils.data._utils.collate
_default_collate_fn_map[DGLGraph] = collate_dgl_graph_fn

_default_collate_fn_map[SmilesList] = collate_smiles_list


def default_collate(batch):
    return collate(batch, collate_fn_map=_default_collate_fn_map)
