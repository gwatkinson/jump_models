# Inspired by https://github.com/HannesStark/3DInfomax/blob/master/models/pna.py

from functools import partial
from math import sqrt
from typing import Callable, Dict, List, Optional, Union

import dgl
import numpy as np
import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch import nn

from src.modules.layers.pna_layers import MLP

EPS = 1e-5


full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()


def aggregate_mean(h, **kwargs):
    return torch.mean(h, dim=-2)


def aggregate_max(h, **kwargs):
    return torch.max(h, dim=-2)[0]


def aggregate_min(h, **kwargs):
    return torch.min(h, dim=-2)[0]


def aggregate_std(h, **kwargs):
    return torch.sqrt(aggregate_var(h) + EPS)


def aggregate_var(h, **kwargs):
    h_mean_squares = torch.mean(h * h, dim=-2)
    h_mean = torch.mean(h, dim=-2)
    var = torch.relu(h_mean_squares - h_mean * h_mean)
    return var


def aggregate_moment(h, n=3, **kwargs):
    # for each node (E[(X-E[X])^n])^{1/n}
    # EPS is added to the absolute value of expectation before taking the nth root for stability
    h_mean = torch.mean(h, dim=-2, keepdim=True)
    h_n = torch.mean(torch.pow(h - h_mean, n), dim=-2)
    rooted_h_n = torch.sign(h_n) * torch.pow(torch.abs(h_n) + EPS, 1.0 / n)
    return rooted_h_n


def aggregate_sum(h, **kwargs):
    return torch.sum(h, dim=-2)


# each scaler is a function that takes as input X (B x N x Din), adj (B x N x N) and
# avg_d (dictionary containing averages over training set) and returns X_scaled (B x N x Din) as output


def scale_identity(h, D=None, avg_d=None):
    return h


def scale_amplification(h, D, avg_d):
    # log(D + 1) / d * h     where d is the average of the ``log(D + 1)`` in the training set
    return h * (np.log(D + 1) / avg_d["log"])


def scale_attenuation(h, D, avg_d):
    # (log(D + 1))^-1 / d * X     where d is the average of the ``log(D + 1))^-1`` in the training set
    return h * (avg_d["log"] / np.log(D + 1))


PNA_AGGREGATORS = {
    "mean": aggregate_mean,
    "sum": aggregate_sum,
    "max": aggregate_max,
    "min": aggregate_min,
    "std": aggregate_std,
    "var": aggregate_var,
    "moment3": partial(aggregate_moment, n=3),
    "moment4": partial(aggregate_moment, n=4),
    "moment5": partial(aggregate_moment, n=5),
}

PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super().__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for _, dim in enumerate(full_atom_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def reset_parameters(self):
        for _, embedder in enumerate(self.atom_embedding_list):
            embedder.weight.data.uniform_(-sqrt(3), sqrt(3))

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            if self.padding:
                x_embedding += self.atom_embedding_list[i](x[:, i] + 1)
            else:
                x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim, padding=False):
        """
        :param emb_dim: the dimension that the returned embedding will have
        :param padding: if this is true then -1 will be mapped to padding
        """
        super().__init__()

        self.bond_embedding_list = torch.nn.ModuleList()
        self.padding = padding

        for _, dim in enumerate(full_bond_feature_dims):
            if padding:
                emb = torch.nn.Embedding(dim + 1, emb_dim, padding_idx=0)
            else:
                emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if self.padding:
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i] + 1)
            else:
                bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding


class PNA(nn.Module):
    """Message Passing Neural Network that does not use 3D information."""

    def __init__(
        self,
        hidden_dim,
        target_dim,
        out_dim,
        aggregators: List[str],
        scalers: List[str],
        readout_aggregators: List[str],
        ckpt_path: Optional[str] = None,
        readout_batchnorm: bool = True,
        readout_hidden_dim=None,
        readout_layers: int = 2,
        residual: bool = True,
        pairwise_distances: bool = False,
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        propagation_depth: int = 5,
        dropout: float = 0.0,
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        batch_norm_momentum=0.1,
        **kwargs,
    ):
        super().__init__()
        self.node_gnn = PNAGNN(
            hidden_dim=hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            residual=residual,
            pairwise_distances=pairwise_distances,
            activation=activation,
            last_activation=last_activation,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            propagation_depth=propagation_depth,
            dropout=dropout,
            posttrans_layers=posttrans_layers,
            pretrans_layers=pretrans_layers,
            batch_norm_momentum=batch_norm_momentum,
        )
        if readout_hidden_dim is None:
            readout_hidden_dim = hidden_dim
        self.readout_aggregators = readout_aggregators
        self.output = MLP(
            in_dim=hidden_dim * len(self.readout_aggregators),
            hidden_size=readout_hidden_dim,
            mid_batch_norm=readout_batchnorm,
            out_dim=target_dim,
            layers=readout_layers,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.out_dim = out_dim

        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))

            # OGB added a new dim for the atom embedding of chirality. I duplicate the last dimension of the ckpt to match the new dim.
            # see https://github.com/snap-stanford/ogb/blame/745531be13c5403a93c80e21a41848e38ea7637c/ogb/utils/features.py#L8
            # TODO: load the correct config file (ie the one used for training)
            if ckpt["model_state_dict"]["node_gnn.atom_encoder.atom_embedding_list.1.weight"].shape[0] != 5:
                ckpt["model_state_dict"]["node_gnn.atom_encoder.atom_embedding_list.1.weight"] = torch.cat(
                    (
                        ckpt["model_state_dict"]["node_gnn.atom_encoder.atom_embedding_list.1.weight"],
                        ckpt["model_state_dict"]["node_gnn.atom_encoder.atom_embedding_list.1.weight"][-1, None, :],
                    )
                )

            self.load_state_dict(ckpt["model_state_dict"])

        self.backbone = self.node_gnn

        self.projection_head = nn.Sequential(
            nn.Linear(target_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(0.1),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, graph: dgl.DGLGraph):
        self.node_gnn(graph)
        readouts_to_cat = [dgl.readout_nodes(graph, "feat", op=aggr) for aggr in self.readout_aggregators]
        readout = torch.cat(readouts_to_cat, dim=-1)
        emb = self.output(readout)
        return self.projection_head(emb)


class PNAGNN(nn.Module):
    def __init__(
        self,
        hidden_dim,
        aggregators: List[str],
        scalers: List[str],
        residual: bool = True,
        pairwise_distances: bool = False,
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum=0.1,
        propagation_depth: int = 5,
        dropout: float = 0.0,
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.mp_layers = nn.ModuleList()

        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(
                    in_dim=hidden_dim,
                    out_dim=int(hidden_dim),
                    in_dim_edges=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    pairwise_distances=pairwise_distances,
                    residual=residual,
                    dropout=dropout,
                    activation=activation,
                    last_activation=last_activation,
                    mid_batch_norm=mid_batch_norm,
                    last_batch_norm=last_batch_norm,
                    avg_d={"log": 1.0},
                    posttrans_layers=posttrans_layers,
                    pretrans_layers=pretrans_layers,
                    batch_norm_momentum=batch_norm_momentum,
                ),
            )
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

    def forward(self, graph: dgl.DGLGraph):
        graph.ndata["feat"] = self.atom_encoder(graph.ndata["feat"])
        graph.edata["feat"] = self.bond_encoder(graph.edata["feat"])

        for mp_layer in self.mp_layers:
            mp_layer(graph)


AVG_D = {"log": 1.0}


class PNALayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        aggregators: List[str],
        scalers: List[str],
        activation: Union[Callable, str] = "relu",
        last_activation: Union[Callable, str] = "none",
        dropout: float = 0.0,
        residual: bool = True,
        pairwise_distances: bool = False,
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum=0.1,
        avg_d: Dict[str, float] = AVG_D,
        posttrans_layers: int = 2,
        pretrans_layers: int = 1,
    ):
        super().__init__()
        self.aggregators = [PNA_AGGREGATORS[aggr] for aggr in aggregators]
        self.scalers = [PNA_SCALERS[scale] for scale in scalers]
        self.edge_features = in_dim_edges > 0
        self.activation = activation
        self.avg_d = avg_d
        self.pairwise_distances = pairwise_distances
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges + 1) if self.pairwise_distances else (2 * in_dim + in_dim_edges),
            hidden_size=in_dim,
            out_dim=in_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=pretrans_layers,
            mid_activation=activation,
            dropout=dropout,
            last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.posttrans = MLP(
            in_dim=(len(self.aggregators) * len(self.scalers) + 1) * in_dim,
            hidden_size=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=activation,
            last_activation=last_activation,
            dropout=dropout,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            batch_norm_momentum=batch_norm_momentum,
        )

    def forward(self, g):
        h = g.ndata["feat"]
        h_in = h
        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata["feat"]], dim=-1)
        # post-transformation
        h = self.posttrans(h)
        if self.residual:
            h = h + h_in

        g.ndata["feat"] = h

    def message_func(self, edges) -> Dict[str, torch.Tensor]:
        r"""The message function to generate messages along the edges."""
        return {"e": edges.data["e"]}

    def reduce_func(self, nodes) -> Dict[str, torch.Tensor]:
        r"""The reduce function to aggregate the messages.

        Apply the aggregators and scalers, and concatenate the results.
        """
        h_in = nodes.data["feat"]
        h = nodes.mailbox["e"]
        D = h.shape[-2]
        h_to_cat = [aggr(h=h, h_in=h_in) for aggr in self.aggregators]
        h = torch.cat(h_to_cat, dim=-1)

        if len(self.scalers) > 1:
            h = torch.cat([scale(h, D=D, avg_d=self.avg_d) for scale in self.scalers], dim=-1)

        return {"feat": h}

    def pretrans_edges(self, edges) -> Dict[str, torch.Tensor]:
        r"""Return a mapping to the concatenation of the features from the
        source node, the destination node, and the edge between them (if
        applicable)."""
        if self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src["x"] - edges.dst["x"]) ** 2, dim=-1)[:, None]
            z2 = torch.cat([edges.src["feat"], edges.dst["feat"], edges.data["feat"], squared_distance], dim=-1)
        elif not self.edge_features and self.pairwise_distances:
            squared_distance = torch.sum((edges.src["x"] - edges.dst["x"]) ** 2, dim=-1)[:, None]
            z2 = torch.cat([edges.src["feat"], edges.dst["feat"], squared_distance], dim=-1)
        elif self.edge_features and not self.pairwise_distances:
            z2 = torch.cat([edges.src["feat"], edges.dst["feat"], edges.data["feat"]], dim=-1)
        else:
            z2 = torch.cat([edges.src["feat"], edges.dst["feat"]], dim=-1)
        return {"e": self.pretrans(z2)}
