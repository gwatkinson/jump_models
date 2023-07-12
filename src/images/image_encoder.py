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
        self.target_num = target_num

        self.model = timm.create_model(model_name, pretrained=self.pretrained)

        if ("efficientnet" in model_name) or ("mixnet" in model_name):
            self.model.conv_stem.weight = nn.Parameter(
                self.model.conv_stem.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif model_name in ["resnet34d"]:
            self.model.conv1[0].weight = nn.Parameter(
                self.model.conv1[0].weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif ("resnet" in model_name or "resnest" in model_name) and "vit" not in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif "rexnet" in model_name or "regnety" in model_name or "nf_regnet" in model_name:
            self.model.stem.conv.weight = nn.Parameter(
                self.model.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif "resnext" in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.fc.in_features, target_num)
            self.model.fc = nn.Identity()
        elif "hrnet_w32" in model_name:
            self.model.conv1.weight = nn.Parameter(self.model.conv1.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch])
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif "densenet" in model_name:
            self.model.features.conv0.weight = nn.Parameter(
                self.model.features.conv0.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.classifier.in_features, target_num)
            self.model.classifier = nn.Identity()
        elif "ese_vovnet39b" in model_name or "xception41" in model_name:
            self.model.stem[0].conv.weight = nn.Parameter(
                self.model.stem[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.head.fc.in_features, target_num)
            self.model.head.fc = nn.Identity()
        elif "dpn" in model_name:
            self.model.features.conv1_1.conv.weight = nn.Parameter(
                self.model.features.conv1_1.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.classifier.in_channels, target_num)
            self.model.classifier = nn.Identity()
        elif "inception" in model_name:
            self.model.features[0].conv.weight = nn.Parameter(
                self.model.features[0].conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.last_linear.in_features, target_num)
            self.model.last_linear = nn.Identity()
        elif "vit" in model_name:
            self.model.patch_embed.proj.weight = nn.Parameter(
                self.model.patch_embed.proj.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        elif "vit_base_resnet50" in model_name:
            self.model.patch_embed.backbone.stem.conv.weight = nn.Parameter(
                self.model.patch_embed.backbone.stem.conv.weight.repeat(1, n_ch // 3 + 1, 1, 1)[:, :n_ch]
            )
            self.myfc = nn.Linear(self.model.head.in_features, target_num)
            self.model.head = nn.Identity()
        else:
            raise

        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

    def extract(self, x):
        return self.model(x)

    def forward(self, x):
        x = self.extract(x)
        re = self.myfc(x)
        return re

    def __call__(self, x):
        return self.model(x)
