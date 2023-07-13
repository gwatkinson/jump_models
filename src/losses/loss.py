from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F


def symmetric_loss_wrapper(func):
    @wraps(func)
    def make_symmetric_func(in_1, in_2, *args, **kwargs):
        a = func(in_1, in_2, *args, **kwargs)
        b = func(in_2, in_1, *args, **kwargs)
        return a + b / 2.0

    return make_symmetric_func


@symmetric_loss_wrapper
def dualModalityInfoNCE_loss(X, Y, temperature, normalize=False):
    if normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = Y.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, temperature)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)

    return CL_loss


@symmetric_loss_wrapper
def cdist_loss(X, Y, margin):
    dist = torch.cdist(X, Y, p=2)
    pos = torch.diag(dist)

    bs = X.size(0)
    mask = torch.eye(bs, device=X.device)
    neg = (1 - mask) * dist + mask * margin
    neg = torch.relu(margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / bs / (bs - 1)
    return loss
