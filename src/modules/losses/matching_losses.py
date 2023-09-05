# Inspired from https://github.com/prokia/MIGA/blob/main/pretrain.py#L54

# flake8: noqa

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

# TODO: Improve the fusion of the two embeddings (instead of simple concatenation)


class GraphImageMatchingLoss(_Loss):
    def __init__(
        norm: bool = True,
        name: str = "GraphImageMatchingLoss",
        **kwargs,
    ):
        super().__init__()
        self.norm = norm
        self.name = name

    def forward(self, graph_emb, img_emb, **kwargs) -> Tensor:
        batch_size, metric_dim = graph_emb.size()

        output_pos = torch.cat([graph_emb, img_emb], dim=1)  # [batch_size, 2 * metric_dim]

        with torch.no_grad():
            weights_g2i = F.softmax(torch.cdist(graph_emb, img_emb, p=2))
            weights_i2g = F.softmax(torch.cdist(img_emb, graph_emb, p=2))
            weights_i2g.fill_diagonal_(0)
            weights_g2i.fill_diagonal_(0)

        # select a negative image for each text
        img_embeds_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_g2i[b], 1).item()
            img_embeds_neg.append(img_emb[neg_idx])
        img_embeds_neg = torch.stack(img_embeds_neg, dim=0)

        # select a negative text for each image
        graph_embeds_neg = []
        for b in range(batch_size):
            neg_idx = torch.multinomial(weights_i2g[b], 1).item()
            graph_embeds_neg.append(graph_emb[neg_idx])
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)

        graph_embeds_all = torch.cat([graph_emb, graph_embeds_neg], dim=0)
        img_embeds_all = torch.cat([img_emb, img_embeds_neg], dim=0)

        output_neg = torch.cat([graph_embeds_all, img_embeds_all], dim=1)

        vl_embeddings = torch.cat([output_pos, output_neg], dim=0)
        vl_output = self.gim_head(vl_embeddings)

        gim_labels = torch.cat(
            [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)], dim=0
        ).to(graph_emb.device)

        # GGIM_loss = GIM_loss + F.cross_entropy(vl_output, gim_labels)  # GIM loss is the Graph image generation loss with AE

        return F.cross_entropy(vl_output, gim_labels)
