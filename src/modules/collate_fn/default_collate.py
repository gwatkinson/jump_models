from dgl import DGLGraph
from dgl import batch as dgl_batch
from torch.utils.data._utils.collate import collate, default_collate_fn_map

default_collate_fn_map[
    DGLGraph
] = dgl_batch  # Add DGLGraph to default_collate_fn_map of torch.utils.data._utils.collate


def default_collate(x):
    return collate(x, collate_fn_map=default_collate_fn_map)
