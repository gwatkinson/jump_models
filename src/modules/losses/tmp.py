class BatchMaskingGraph(Data):
    def __init__(self, batch=None, **kwargs):
        super().__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(items, transforms=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        if isinstance(items[0], tuple):
            has_img = True
            data_list = [item[0] for item in items]
            img_list = [item[1] for item in items]
        else:
            has_img = False
            data_list = items
            img_list = items

        if transforms is not None:
            if not isinstance(transforms, list):
                transforms = [transforms]
            for transform in transforms:
                data_list = [transform(d) for d in data_list]

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert "batch" not in keys

        batch = BatchMaskingGraph()

        # print(data_list[0])
        # keys = ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct", "overlap_context_substruct_idx", "edge_attr_context", "edge_index_context", "x_context", 'x', 'edge_attr', 'edge_index', "mask_node_idx", 'mask_node_label', 'mask_edge_idx', 'mask_edge_label']
        # keys = ['edge_index_substruct', 'x', 'edge_index', 'edge_attr_substruct', 'x_context', 'overlap_context_substruct_idx', 'mask_node_label', 'edge_attr', 'masked_atom_indices', 'center_substruct_idx', 'x_substruct', 'edge_attr_context', 'masked_x', 'edge_index_context']
        # print(keys)
        for key in keys:
            batch[key] = []
        batch.batch = []
        # used for pooling the context
        batch.batch_overlapped_context = []
        batch.overlapped_context_size = []

        cumsum_node = 0
        cumsum_edge = 0
        cumsum_main = 0
        cumsum_substruct = 0
        cumsum_context = 0

        i = 0
        imgs = []
        for data, img in zip(data_list, img_list):
            # If there is no context, just skip!!
            if hasattr(data, "x_context"):
                num_nodes = data.num_nodes
                num_nodes_substruct = len(data.x_substruct)
                num_nodes_context = len(data.x_context)

                batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
                batch.batch_overlapped_context.append(
                    torch.full((len(data.overlap_context_substruct_idx),), i, dtype=torch.long)
                )
                batch.overlapped_context_size.append(len(data.overlap_context_substruct_idx))

                ##batching for the main graph
                for key in data.keys:
                    # for key in ['x', 'edge_attr', 'edge_index']:

                    # if not "context" in key and not "substruct" in key:
                    if key in ["x", "edge_attr", "edge_index"]:
                        item = data[key]
                        # item = item + cumsum_main if batch.cumsum(key, item) else item
                        if key in ["edge_index"]:
                            item = item + cumsum_main
                        batch[key].append(item)

                    ###batching for the substructure graph
                    elif key in ["center_substruct_idx", "edge_attr_substruct", "edge_index_substruct", "x_substruct"]:
                        item = data[key]
                        item = item + cumsum_substruct if batch.cumsum(key, item) else item
                        batch[key].append(item)

                    ###batching for the context graph
                    elif key in [
                        "overlap_context_substruct_idx",
                        "edge_attr_context",
                        "edge_index_context",
                        "x_context",
                    ]:
                        item = data[key]
                        item = item + cumsum_context if batch.cumsum(key, item) else item
                        batch[key].append(item)

                    elif key in ["masked_atom_indices"]:
                        item = data[key]
                        item = item + cumsum_node
                        batch[key].append(item)

                    elif key == "connected_edge_indices":
                        item = data[key]
                        item = item + cumsum_edge
                        batch[key].append(item)

                    else:
                        item = data[key]
                        batch[key].append(item)

                cumsum_node += num_nodes
                cumsum_edge += data.edge_index.shape[1]
                cumsum_main += num_nodes
                cumsum_substruct += num_nodes_substruct
                cumsum_context += num_nodes_context
                i += 1

                imgs.append(img)

        for key in keys:
            batch[key] = torch.cat(batch[key], dim=batch.cat_dim(key))
        batch.batch = torch.cat(batch.batch, dim=-1)
        batch.batch_overlapped_context = torch.cat(batch.batch_overlapped_context, dim=-1)
        batch.overlapped_context_size = torch.LongTensor(batch.overlapped_context_size)

        if has_img:
            imgs_batch = torch.stack(imgs, 0)
            return batch.contiguous(), imgs_batch
        else:
            return batch.contiguous()

    def cat_dim(self, key):
        return -1 if key in ["edge_index", "edge_index_substruct", "edge_index_context"] else 0

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.

        .. note::     This method is for internal use only, and should
        only be overridden     if the batch concatenation process is
        corrupted for a specific data     attribute.
        """
        return key in [
            "edge_index",
            "edge_index_substruct",
            "edge_index_context",
            "overlap_context_substruct_idx",
            "center_substruct_idx",
        ]

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1
