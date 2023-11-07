import timm
import torch.nn as nn

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


def replace_dropout(module, name, dropout=0.2, name_to_change="drop_block"):
    """Recursively put desired batch norm in nn.module module.

    set module = net to start code.
    """
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        if name_to_change in attr_str:
            logger.debug("replaced: ", name, attr_str)
            new_dropout = nn.Dropout(dropout)
            setattr(module, attr_str, new_dropout)

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_dropout(immediate_child_module, name, dropout=dropout, name_to_change=name_to_change)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        instance_model_name: str,
        n_channels: int = 5,
        pretrained: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_channels = n_ch = n_channels
        self.pretrained = pretrained
        self.model_name = model_name = instance_model_name
        self.dropout = dropout

        self.backbone = timm.create_model(model_name, pretrained=self.pretrained)

        if ("efficientnet" in model_name) or ("mixnet" in model_name):
            self.backbone.conv_stem.weight = nn.Parameter(
                self.backbone.conv_stem.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )

            self.out_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif model_name in ["resnet34d"]:
            self.backbone.conv1[0].weight = nn.Parameter(
                self.backbone.conv1[0].weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif ("resnet" in model_name or "resnest" in model_name) and "vit" not in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

            if dropout > 0.0:
                logger.info(f"Setting dropout rate to {dropout}")
                replace_dropout(self.backbone, name="backbone", dropout=dropout, name_to_change="drop_block")

        elif "rexnet" in model_name or "regnety" in model_name or "nf_regnet" in model_name:
            self.backbone.stem.conv.weight = nn.Parameter(
                self.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.stem.conv
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.head.fc.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        elif "resnext" in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.conv1
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.fc.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif "hrnet_w32" in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.conv1
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.classifier.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif "densenet" in model_name:
            self.backbone.features.conv0.weight = nn.Parameter(
                self.backbone.features.conv0.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.features.conv0
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.classifier.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif "ese_vovnet39b" in model_name or "xception41" in model_name:
            self.backbone.stem[0].conv.weight = nn.Parameter(
                self.backbone.stem[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.stem[0].conv
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.head.fc.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()

        elif "dpn" in model_name:
            self.backbone.features.conv1_1.conv.weight = nn.Parameter(
                self.backbone.features.conv1_1.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.features.conv1_1.conv
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.classifier.in_channels, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.classifier.in_channels
            self.backbone.classifier = nn.Identity()

        elif "inception" in model_name:
            self.backbone.features[0].conv.weight = nn.Parameter(
                self.backbone.features[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.features[0].conv
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.last_linear.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.last_linear.in_features
            self.backbone.last_linear = nn.Identity()

        elif "vit_base_resnet50" in model_name or "vit_base_r50" in model_name:
            self.backbone.patch_embed.backbone.stem.conv.weight = nn.Parameter(
                self.backbone.patch_embed.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.patch_embed.backbone.stem.conv
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.head.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        elif "vit" in model_name:
            self.backbone.patch_embed.proj.weight = nn.Parameter(
                self.backbone.patch_embed.proj.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            # self.entry = self.backbone.patch_embed.proj
            # self.projection_head = nn.Sequential(
            #     nn.Linear(self.backbone.head.in_features, out_dim),
            #     nn.ReLU(),
            #     nn.Linear(out_dim, out_dim),
            # )
            self.out_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()

        else:
            raise

        # self.backbone.to("cpu")
        # self.projection_head.to("cpu")

        # self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def extract(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        # re = self.projection_head(x)
        return x
