import dgl
import torch


def label_graph_collate_function(data):
    """Collate function for batching DGLGraphs and labels.

    Args:
        data: list of dicts with keys 'compound' and 'label'

    Returns:
        dict with keys 'compound' and 'label'
    """
    compound = dgl.batch([d["compound"] for d in data])
    label = torch.stack([d["label"] for d in data])

    return {"compound": compound, "label": label}
