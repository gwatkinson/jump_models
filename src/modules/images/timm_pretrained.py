import timm
import torch.nn as nn


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
            self.projection_head = nn.Linear(self.backbone.classifier.in_features, out_dim)
            self.backbone.classifier = nn.Identity()
        elif model_name in ["resnet34d"]:
            self.backbone.conv1[0].weight = nn.Parameter(
                self.backbone.conv1[0].weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.fc.in_features, out_dim)
            self.backbone.fc = nn.Identity()
        elif ("resnet" in model_name or "resnest" in model_name) and "vit" not in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.fc.in_features, out_dim)
            self.backbone.fc = nn.Identity()
        elif "rexnet" in model_name or "regnety" in model_name or "nf_regnet" in model_name:
            self.backbone.stem.conv.weight = nn.Parameter(
                self.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.head.fc.in_features, out_dim)
            self.backbone.head.fc = nn.Identity()
        elif "resnext" in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.fc.in_features, out_dim)
            self.backbone.fc = nn.Identity()
        elif "hrnet_w32" in model_name:
            self.backbone.conv1.weight = nn.Parameter(
                self.backbone.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.classifier.in_features, out_dim)
            self.backbone.classifier = nn.Identity()
        elif "densenet" in model_name:
            self.backbone.features.conv0.weight = nn.Parameter(
                self.backbone.features.conv0.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.classifier.in_features, out_dim)
            self.backbone.classifier = nn.Identity()
        elif "ese_vovnet39b" in model_name or "xception41" in model_name:
            self.backbone.stem[0].conv.weight = nn.Parameter(
                self.backbone.stem[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.head.fc.in_features, out_dim)
            self.backbone.head.fc = nn.Identity()
        elif "dpn" in model_name:
            self.backbone.features.conv1_1.conv.weight = nn.Parameter(
                self.backbone.features.conv1_1.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.classifier.in_channels, out_dim)
            self.backbone.classifier = nn.Identity()
        elif "inception" in model_name:
            self.backbone.features[0].conv.weight = nn.Parameter(
                self.backbone.features[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.last_linear.in_features, out_dim)
            self.backbone.last_linear = nn.Identity()
        elif "vit" in model_name:
            self.backbone.patch_embed.proj.weight = nn.Parameter(
                self.backbone.patch_embed.proj.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.head.in_features, out_dim)
            self.backbone.head = nn.Identity()
        elif "vit_base_resnet50" in model_name:
            self.backbone.patch_embed.backbone.stem.conv.weight = nn.Parameter(
                self.backbone.patch_embed.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.projection_head = nn.Linear(self.backbone.head.in_features, out_dim)
            self.backbone.head = nn.Identity()
        else:
            raise

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def extract(self, x):
        return self.backbone(x)

    def forward(self, x):
        x = self.backbone(x)
        re = self.projection_head(x)
        return re
