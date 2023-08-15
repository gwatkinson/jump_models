import timm
import torch.nn as nn

from src.utils import pylogger

logger = pylogger.get_pylogger(__name__)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        instance_model_name: str,
        target_num: int,
        n_channels: int = 5,
        pretrained: bool = True,
    ):
        super().__init__()

        self.n_channels = n_ch = n_channels
        self.pretrained = pretrained
        self.model_name = model_name = instance_model_name
        self.out_dim = out_dim = target_num

        self.backbone = timm.create_model(model_name, pretrained=self.pretrained)

        if ("efficientnet" in model_name) or ("mixnet" in model_name):
            self.backbone.conv_stem.weight = nn.Parameter(
                self.backbone.conv_stem.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.conv_stem
            self.projection_head = nn.Linear(self.backbone.classifier.in_features, out_dim)
            self.backbone.classifier = nn.Identity()
            logger.info("Using model efficientnet/mixnet with projection head")
        elif model_name in ["resnet34d"]:
            self.backbone.conv1[0].weight = nn.Parameter(
                self.backbone.conv1[0].weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.conv1[0]
            self.projection_head = nn.Linear(self.backbone.fc.in_features, out_dim)
            self.backbone.fc = nn.Identity()
            logger.info("Using model resnet34d with projection head")
        elif ("resnet" in model_name or "resnest" in model_name) and "vit" not in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.conv1
            self.projection_head = nn.Linear(self.backbone.fc.in_features, out_dim)
            self.backbone.fc = nn.Identity()
            logger.info("Using model resnet/resnest with projection head")
        elif "rexnet" in model_name or "regnety" in model_name or "nf_regnet" in model_name:
            self.backbone.stem.conv.weight = nn.Parameter(
                self.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.stem.conv
            self.projection_head = nn.Linear(self.backbone.head.fc.in_features, out_dim)
            self.backbone.head.fc = nn.Identity()
            logger.info("Using model rexnet/regnety/nf_regnet with projection head")
        elif "resnext" in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.conv1
            self.projection_head = nn.Linear(self.backbone.fc.in_features, out_dim)
            self.backbone.fc = nn.Identity()
            logger.info("Using model resnext with projection head")
        elif "hrnet_w32" in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.conv1
            self.projection_head = nn.Linear(self.backbone.classifier.in_features, out_dim)
            self.backbone.classifier = nn.Identity()
            logger.info("Using model hrnet_w32 with projection head")
        elif "densenet" in model_name:
            self.backbone.features.conv0.weight = nn.Parameter(
                self.backbone.features.conv0.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.features.conv0
            self.projection_head = nn.Linear(self.backbone.classifier.in_features, out_dim)
            self.backbone.classifier = nn.Identity()
            logger.info("Using model densenet with projection head")
        elif "ese_vovnet39b" in model_name or "xception41" in model_name:
            self.backbone.stem[0].conv.weight = nn.Parameter(
                self.backbone.stem[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.stem[0].conv
            self.projection_head = nn.Linear(self.backbone.head.fc.in_features, out_dim)
            self.backbone.head.fc = nn.Identity()
            logger.info("Using model ese_vovnet39b/xception41 with projection head")
        elif "dpn" in model_name:
            self.backbone.features.conv1_1.conv.weight = nn.Parameter(
                self.backbone.features.conv1_1.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.features.conv1_1.conv
            self.projection_head = nn.Linear(self.backbone.classifier.in_channels, out_dim)
            self.backbone.classifier = nn.Identity()
            logger.info("Using model dpn with projection head")
        elif "inception" in model_name:
            self.backbone.features[0].conv.weight = nn.Parameter(
                self.backbone.features[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.features[0].conv
            self.projection_head = nn.Linear(self.backbone.last_linear.in_features, out_dim)
            self.backbone.last_linear = nn.Identity()
            logger.info("Using model inception with projection head")
        elif "vit_base_resnet50" in model_name or "vit_base_r50" in model_name:
            self.backbone.patch_embed.backbone.stem.conv.weight = nn.Parameter(
                self.backbone.patch_embed.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.patch_embed.backbone.stem.conv
            self.projection_head = nn.Linear(self.backbone.head.in_features, out_dim)
            self.backbone.head = nn.Identity()
            logger.info("Using model vit_base_resnet50 with projection head")
        elif "vit" in model_name:
            self.backbone.patch_embed.proj.weight = nn.Parameter(
                self.backbone.patch_embed.proj.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.entry = self.backbone.patch_embed.proj
            self.projection_head = nn.Linear(self.backbone.head.in_features, out_dim)
            self.backbone.head = nn.Identity()
            logger.info("Using model vit with projection head")
        else:
            raise

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def extract(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        re = self.projection_head(x)
        return re
