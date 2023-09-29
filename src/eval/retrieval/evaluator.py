from collections import defaultdict
from typing import Optional

import numpy as np
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

    def visualize(self, outs, **kwargs):
        pass

    def run(self):
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="predict")
        self.model.eval()

        predict_loaders = self.datamodule.predict_dataloader()

        out_metrics = defaultdict(lambda: defaultdict(list))
        for gene in predict_loaders:
            # List of the embeddings of each group -> Ng x 120 x E
            compound_embs = self.trainer.predict(self.model, predict_loaders[gene]["molecule"])

            image_emb = self.trainer.predict(self.model, predict_loaders[gene]["image"])
            # Only needed if more images than batch size (rare) -> Ni x E
            image_emb = concat_from_list_of_dict(image_emb, "image")

            for group in compound_embs:
                activities = group["activity_flag"]
                compound_emb = group["compound"]

                gene_group_metrics = self.model.retrieval(
                    image_embeddings=image_emb, compound_embeddings=compound_emb, activities=activities
                )

                for metric in gene_group_metrics:
                    out_metrics[gene][metric].append(gene_group_metrics[metric])

        aggregate_metrics = defaultdict(lambda: 0)
        for gene in out_metrics:
            for metric in out_metrics[gene]:
                aggregate_metrics[f"{gene}/{metric}_mean"] = np.mean(out_metrics[gene][metric])
                aggregate_metrics[f"{gene}/{metric}_std"] = np.std(out_metrics[gene][metric])

        num_genes = len(out_metrics)
        for gene in out_metrics:
            for metric in out_metrics[gene]:
                aggregate_metrics[f"Average/{metric}"] += aggregate_metrics[f"{gene}/{metric}_mean"] / num_genes

        aggregate_metrics = dict(aggregate_metrics)

        for logger in self.trainer.loggers:
            logger.log_metrics(aggregate_metrics)
            logger.save()

        return aggregate_metrics
