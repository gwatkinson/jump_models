from collections import defaultdict
from typing import Optional

import numpy as np
from lightning import LightningDataModule, LightningModule, Trainer

from src.eval import Evaluator
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


def concat_from_list_of_dict(res, key):
    out = np.concatenate([r[key] for r in res])
    if out.ndim == 2:
        return out.tolist()
    else:
        return out


class SimpleRetrievalEvaluator(Evaluator):
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

    def run(self):
        predictions = self.trainer.predict(self.model, self.datamodule)
        keys = list(predictions[0].keys())
        return {key: concat_from_list_of_dict(predictions, key) for key in keys}

    def evaluate(self):
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="predict")
        self.model.eval()

        predict_loaders = self.datamodule.predict_dataloader()

        out_metrics = defaultdict(lambda: defaultdict(lambda: 0))
        for gene in predict_loaders:
            # List of the embeddings of each group -> Ng x 120 x E
            compound_embs = self.trainer.predict(self.model, predict_loaders[gene]["molecule"])

            # activities = concat_from_list_of_dict(compound_emb, "activity_flag")
            # compound_emb = concat_from_list_of_dict(compound_emb, "compound")

            image_emb = self.trainer.predict(self.model, predict_loaders[gene]["image"])
            # Only needed if more images than batch size (rare) -> Ni x E
            image_emb = concat_from_list_of_dict(image_emb, "image")

            num_groups = len(compound_embs)
            for group in compound_embs:
                activities = group["activity_flag"]
                compound_emb = group["compound"]

                gene_group_metrics = self.model.retrieval(
                    image_embeddings=image_emb, compound_embeddings=compound_emb, activities=activities
                )

                for metric in gene_group_metrics:
                    out_metrics[gene][metric] += gene_group_metrics[metric] / num_groups

        out_metrics = dict(out_metrics)

        mean_metrics = defaultdict(lambda: 0)
        num_genes = len(out_metrics)
        for gene in out_metrics:
            for metric in out_metrics[gene]:
                mean_metrics[metric] += out_metrics[gene][metric] / num_genes
        mean_metrics = dict(mean_metrics)

        out_metrics["Average"] = mean_metrics

        unfold_metrics = {}
        for gene in out_metrics:
            for metric in out_metrics[gene]:
                unfold_metrics[f"{gene}/{metric}"] = out_metrics[gene][metric]

        for logger in self.trainer.loggers:
            logger.log_metrics(unfold_metrics)
            logger.save()

        return unfold_metrics
