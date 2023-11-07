import dgl
import torch


def idr_flag_graph_collate_fn(data):
    compound = dgl.batch([d["compound"] for d in data])
    label = torch.Tensor([d["activity_flag"] for d in data])

    return {"compound": compound, "activity_flag": label}


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


def image_graph_collate_function(data):
    """Collate function for the MoleculeImageDataset.

    Args:
        data: list of dicts with keys 'image' and 'compound'

    Returns:
        dict with keys 'image' and 'compound'
    """
    image = torch.stack([d["image"] for d in data])
    compound = dgl.batch([d["compound"] for d in data])

    return {"image": image, "compound": compound}


def image_graph_label_collate_function(data):
    """Collate function for image/graph/label data.

    Args:
        data: list of dicts with keys 'label', 'image' and 'compound'

    Returns:
        dict with keys 'label', 'image' and 'compound'
    """
    labels = torch.Tensor([d["label"] for d in data]).long()
    image = torch.stack([d["image"] for d in data])
    compound = dgl.batch([d["compound"] for d in data])

    return {"label": labels, "image": image, "compound": compound}
