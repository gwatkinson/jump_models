import dgl
import torch


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
