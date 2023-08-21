from dgl import DGLHeteroGraph, batch
from torch.utils.data._utils.collate import collate, default_collate_fn_map

default_collate_fn_map[
    DGLHeteroGraph
] = batch  # Add DGLGraph to default_collate_fn_map of torch.utils.data._utils.collate


def default_collate(batch):
    return collate(batch, collate_fn_map=default_collate_fn_map)
