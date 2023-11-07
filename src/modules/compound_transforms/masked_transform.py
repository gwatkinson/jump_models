import random
from typing import Any

import dgl
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data


def dgl_to_pyg_graph(
    g: Any,
) -> Data:
    data = Data()
    data.edge_index = torch.stack(g.edges(), dim=0)

    if len(g.ndata) == 1:
        data["x"] = g.ndata[list(g.ndata.keys())[0]]
    else:
        for attr, value in g.ndata.items():
            data[f"n_{attr}"] = value

    if len(g.edata) == 1:
        data["edge_attr"] = g.edata[list(g.edata.keys())[0]]
    else:
        for attr, value in g.edata.items():
            data[f"e_{attr}"] = value

    return data


# atomic_num_idx, chirality_idx, degree, formal_charge, numH, radical_e, hybridization, is_aromatic, is_ring
# bond_type, bond_stereo, is_conjugated


def graph_data_obj_to_nx_clf(data):
    """Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features, and
    represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.

    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i, :2]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j, :2]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx, bond_dir_idx=bond_dir_idx)

    return G


def graph_data_obj_to_nx_reg(data):
    """torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object"""
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        temp_feature = atom_features[i]
        G.add_node(
            i,
            x0=temp_feature[0],
            x1=temp_feature[1],
            x2=temp_feature[2],
            x3=temp_feature[3],
            x4=temp_feature[4],
            x5=temp_feature[5],
            x6=temp_feature[6],
            x7=temp_feature[7],
            x8=temp_feature[8],
        )
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        temp_feature = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, e0=temp_feature[0], e1=temp_feature[1], e2=temp_feature[2])

    return G


def nx_to_graph_data_obj_reg(G):
    """vice versa of graph_data_obj_to_nx_simple()

    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered.
    """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [
            node["x0"],
            node["x1"],
            node["x2"],
            node["x3"],
            node["x4"],
            node["x5"],
            node["x6"],
            node["x7"],
            node["x8"],
        ]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge["e0"], edge["e1"], edge["e2"]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def nx_to_graph_data_obj_clf(G):
    """Converts nx graph to pytorch geometric Data object. Assume node indices.

    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node["atom_num_idx"], node["chirality_tag_idx"]]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge["bond_type_idx"], edge["bond_dir_idx"]]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type The mask edge type
        index in num_possible_edge_type.

        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, task_type="classification", masked_atom_indices=None):
        """
        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label"""
        atom_feature_len = len(data.x[0])
        edge_feature_len = len(data.edge_attr[0])

        if masked_atom_indices is None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        data.masked_x = data.x.clone()
        for atom_idx in masked_atom_indices:
            if task_type == "classification":
                data.masked_x[atom_idx] = torch.tensor([self.num_atom_type] + [0 for _ in range(atom_feature_len - 1)])
            else:
                data.masked_x[atom_idx] = torch.tensor([self.num_atom_type - 1, 0, 0, 0, 0, 0, 0, 0, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in {u, v} and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::1]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    if task_type == "classification":
                        data.edge_attr[bond_idx] = torch.tensor(
                            [self.num_edge_type] + [0 for _ in range(edge_feature_len - 1)]
                        )
                    else:
                        data.edge_attr[bond_idx] = torch.tensor([self.num_edge_type, 0, 0])

                data.connected_edge_indices = torch.tensor(connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return "{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})".format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type, self.mask_rate, self.mask_edge
        )


def reset_idxes(G):
    """Resets node indices such that they are numbered from 0 to num_nodes - 1
    :return: copy of G with relabelled node indices, mapping"""
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the root node."""
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, task_type="classification", root_idx=None):
        """
        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx"""
        num_atoms = data.x.size()[0]
        if root_idx is None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        if task_type == "classification":
            G = graph_data_obj_to_nx_clf(data)  # same ordering as input data obj
        else:
            G = graph_data_obj_to_nx_reg(data)

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0

            if task_type == "classification":
                substruct_data = nx_to_graph_data_obj_clf(substruct_G)
            else:
                substruct_data = nx_to_graph_data_obj_reg(substruct_G)

            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx, self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0

            if task_type == "classification":
                context_data = nx_to_graph_data_obj_clf(context_G)
            else:
                context_data = nx_to_graph_data_obj_reg(context_G)

            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [
                context_node_map[old_idx] for old_idx in context_substruct_overlap_idxes
            ]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

    def __repr__(self):
        return "{}(k={},l1={}, l2={})".format(self.__class__.__name__, self.k, self.l1, self.l2)
