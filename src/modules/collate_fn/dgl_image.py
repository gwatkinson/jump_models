import dgl
import torch


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
