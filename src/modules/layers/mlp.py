from typing import Callable, List, Optional, Union

import torch.nn as nn

# flake8: noqa: B006


class LinearWithActivation(nn.Sequential):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        layers = [nn.Linear(in_dim, out_dim)]

        if norm_layer is not None:
            layers.append(norm_layer(out_dim))
        if activation_layer is not None:
            layers.append(activation_layer())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        super().__init__(*layers)


class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        embedding_dim: Union[int, List[int]] = [512, 512, 512, 512],
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.0,
    ):
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.embedding_dim = embedding_dim

        layers = []
        in_dim = input_dim
        for dim in embedding_dim:
            layers.append(
                LinearWithActivation(
                    in_dim,
                    dim,
                    activation_layer=activation_layer,
                    norm_layer=norm_layer,
                    dropout=dropout,
                )
            )
            in_dim = dim

        layers.append(
            LinearWithActivation(
                in_dim,
                out_dim,
                activation_layer=None,  # No activation or norm on the last layer
                norm_layer=None,
                dropout=dropout,
            )
        )

        super().__init__(*layers)
