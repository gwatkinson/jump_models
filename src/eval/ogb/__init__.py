from typing import Tuple

from lightning import LightningDataModule, LightningModule

from src.eval.ogb.datamodule import (
    BBBPDataModule,
    EsolDataModule,
    HIVDataModule,
    LipoDataModule,
    Tox21DataModule,
    ToxCastDataModule,
)
from src.eval.ogb.module import BBBPModule, EsolModule, HIVModule, LipoModule, Tox21Module, ToxCastModule

ogb_dict = {
    "bbbp": (BBBPDataModule, BBBPModule),
    "hiv": (HIVDataModule, HIVModule),
    "tox21": (Tox21DataModule, Tox21Module),
    "esol": (EsolDataModule, EsolModule),
    "lipo": (LipoDataModule, LipoModule),
    "toxcast": (ToxCastDataModule, ToxCastModule),
}


def get_ogb_module_and_datamodule(ogb_dataset_name: str) -> Tuple[LightningDataModule, LightningModule]:
    return ogb_dict[ogb_dataset_name]
