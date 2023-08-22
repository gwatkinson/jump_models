from collections import defaultdict
from typing import Optional

import torch
from lightning import LightningDataModule, LightningModule, Trainer

from src.eval import Evaluator
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


def concat_from_list_of_dict(res, key):
    return torch.cat([r[key] for r in res], dim=0)


class IDRRetrievalEvaluator(Evaluator):
    """Evaluator for retrieval tasks."""

    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        name: Optional[str] = None,
        visualize_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer

        self.name = name or self.model.__class__.__name__
        self.prefix = f"({self.name}) " if self.name else ""

        self.visualize_kwargs = visualize_kwargs or {}

    def finetune(self):
        # No finetuning for retrieval tasks
        pass

    def evaluate(self):
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="predict")
        self.model.eval()

        predict_loaders = self.datamodule.predict_dataloader()

        out_metrics = {}
        for gene in predict_loaders:
            compound_emb = self.trainer.predict(self.model, predict_loaders[gene]["molecule"])
            activities = concat_from_list_of_dict(compound_emb, "activity_flag")
            compound_emb = concat_from_list_of_dict(compound_emb, "compound")

            image_emb = self.trainer.predict(self.model, predict_loaders[gene]["image"])
            image_emb = concat_from_list_of_dict(image_emb, "image")

            gene_metrics = self.model.retrieval(
                image_embeddings=image_emb, compound_embeddings=compound_emb, activities=activities
            )

            out_metrics[gene] = gene_metrics

        mean_metrics = defaultdict(lambda: 0)
        for gene in out_metrics:
            for metric in out_metrics[gene]:
                mean_metrics[metric] += out_metrics[gene][metric] / len(out_metrics)
        mean_metrics = dict(mean_metrics)
        out_metrics["Average"] = mean_metrics

        for logger in self.trainer.loggers:
            logger.log_metrics(out_metrics)
            logger.save()

        py_logger.info(f"{self.prefix}Retrieval metrics: {out_metrics}")

        return out_metrics

    def visualize(self, **kwargs):
        pass
