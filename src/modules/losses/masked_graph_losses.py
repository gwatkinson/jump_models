from typing import Any, Optional

# from torch.distributions import MultivariateNormal
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.modules.losses.base_losses import LossWithTemperature, RegWithTemperatureLoss

# def from_dgl(
#     g: Any,
# ) -> "torch_geometric.data.Data":
#     import dgl
#     from torch_geometric.data import Data

#     data = Data()
#     data.edge_index = torch.stack(g.edges(), dim=0)

#     for attr, value in g.ndata.items():
#         data[f"n_{attr}"] = value
#     for attr, value in g.edata.items():
#         data[f"e_{attr}"] = value

#     return data


# elif args.MGM_mode == 'MGM':
#     transform1 = MaskAtom(num_atom_type=119, num_edge_type=5,
#                             mask_rate=args.mask_rate, mask_edge=args.mask_edge)

#     l1 = args.num_layer - 1
#     l2 = l1 + args.csize
#     transform2 = ExtractSubstructureContextPair(args.num_layer, l1, l2)

#     loader = DataLoaderMaskingGraph(dataset, transforms=[transform1, transform2], batch_size=args.batch_size,
#                                     shuffle=True, drop_last=True, num_workers=args.num_workers)

# class DataLoaderMaskingGraph(DataLoader):
#     """Data loader which merges data objects from a
#     :class:`torch_geometric.data.dataset` to a mini-batch.
#     Args:
#         dataset (Dataset): The dataset from which to load the data.
#         batch_size (int, optional): How may samples per batch to load.
#             (default: :obj:`1`)
#         shuffle (bool, optional): If set to :obj:`True`, the data will be
#             reshuffled at every epoch (default: :obj:`True`) """

#     def __init__(self, dataset, transforms, batch_size=1, shuffle=True, **kwargs):
#         super(DataLoaderMaskingGraph, self).__init__(
#             dataset,
#             batch_size,
#             shuffle,
#             collate_fn=lambda items: BatchMaskingGraph.from_data_list(items, transforms),
#             **kwargs)


def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim=1)[1] == target).cpu().item()) / len(pred)


def do_AttrMasking(batch, criterion, node_repr, molecule_atom_masking_model):
    target = batch.mask_node_label[:, 0]
    node_pred = molecule_atom_masking_model(node_repr[batch.masked_atom_indices])
    attributemask_loss = criterion(node_pred.double(), target)
    attributemask_acc = compute_accuracy(node_pred, target)
    return attributemask_loss, attributemask_acc


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


def do_ContextPred(
    batch,
    criterion,
    molecule_substruct_model,
    molecule_context_model,
    molecule_readout_func,
    contextpred_neg_samples,
    use_image=True,
    molecule_img_repr=None,
    mol_img_projection_head=None,
    normalize=True,
):
    # creating substructure representation
    substruct_repr = molecule_substruct_model(batch.x_substruct, batch.edge_index_substruct, batch.edge_attr_substruct)[
        batch.center_substruct_idx
    ]

    # creating context representations
    overlapped_node_repr = molecule_context_model(batch.x_context, batch.edge_index_context, batch.edge_attr_context)[
        batch.overlap_context_substruct_idx
    ]

    # positive context representation
    # readout -> global_mean_pool by default
    context_repr = molecule_readout_func(overlapped_node_repr, batch.batch_overlapped_context)

    # Use image embedding
    if molecule_img_repr is not None and use_image:
        if normalize:
            context_repr = F.normalize(context_repr, dim=-1)
            molecule_img_repr = F.normalize(molecule_img_repr, dim=-1)

        context_repr = torch.cat([context_repr, molecule_img_repr], dim=1)
        context_repr = mol_img_projection_head(context_repr)  # reproject to the same dim as context_repr

        # context_repr = 0.8 * context_repr + 0.2 * molecule_img_repr

    # negative contexts are obtained by shifting
    # the indices of context embeddings
    neg_context_repr = torch.cat(
        [context_repr[cycle_index(len(context_repr), i + 1)] for i in range(contextpred_neg_samples)], dim=0
    )

    num_neg = contextpred_neg_samples
    pred_pos = torch.sum(substruct_repr * context_repr, dim=1)
    pred_neg = torch.sum(substruct_repr.repeat((num_neg, 1)) * neg_context_repr, dim=1)

    loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
    loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())

    contextpred_loss = loss_pos + num_neg * loss_neg

    num_pred = len(pred_pos) + len(pred_neg)
    contextpred_acc = (torch.sum(pred_pos > 0).float() + torch.sum(pred_neg < 0).float()) / num_pred
    contextpred_acc = contextpred_acc.detach().cpu().item()

    return contextpred_loss, contextpred_acc


class MaskedGraphModellingLoss(nn.Module):
    def __init__(
        self,
        molecule_gnn_model,
        molecule_readout_func,
        embedding_dim: int,
        num_layer: int,
        csize: int,
        JK: str = "last",
        gnn_type: str = "gin",
        dropout_ratio: float = 0.1,
    ):
        super().__init__()

        self.molecule_gnn_model = molecule_gnn_model
        self.molecule_readout_func = molecule_readout_func

        self.embedding_dim = embedding_dim
        self.num_layer = num_layer
        self.csize = csize
        self.JK = JK
        self.dropout_ratio = dropout_ratio
        self.gnn_type = gnn_type

        self.num_atom = 119

        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = nn.BCEWithLogitsLoss()

        self.molecule_atom_masking_model = torch.nn.Linear(embedding_dim, self.num_atom)

        l1 = num_layer - 1
        l2 = l1 + csize

        self.molecule_context_model = GNN(
            int(l2 - l1), embedding_dim, JK=JK, drop_ratio=dropout_ratio, gnn_type=gnn_type
        )

    def forward(self, batch1, args, criterion):
        masked_node_repr = self.molecule_gnn_model(batch1.masked_x, batch1.edge_index, batch1.edge_attr)

        MGM_loss1, _ = do_AttrMasking(
            batch=batch1,
            criterion=criterion[0],
            node_repr=masked_node_repr,
            molecule_atom_masking_model=self.molecule_atom_masking_model,
        )

        MGM_loss2, _ = do_ContextPred(
            batch=batch1,
            criterion=criterion[1],
            args=args,
            molecule_substruct_model=self.molecule_gnn_model,
            molecule_context_model=self.molecule_context_model,
            molecule_readout_func=self.molecule_readout_func,
            molecule_img_repr=molecule_img_emb,
        )

        MGM_loss = MGM_loss1 + MGM_loss2

        loss_dict = {
            "loss": MGM_loss,
            "MGM_loss1": MGM_loss1,
            "MGM_loss2": MGM_loss2,
        }
