import dgl
import torch


def idr_flag_graph_collate_fn(data):
    compound = dgl.batch([d["compound"] for d in data])
    label = torch.Tensor([d["activity_flag"] for d in data])

    return {"compound": compound, "activity_flag": label}
