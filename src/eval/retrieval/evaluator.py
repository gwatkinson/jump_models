from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from torchmetrics import MetricCollection
from torchmetrics.functional import pairwise_cosine_similarity
from torchmetrics.retrieval import (
    RetrievalFallOut,
    RetrievalHitRate,
    RetrievalMAP,
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)

from src.eval import Evaluator
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


def concat_from_list_of_dict(res, key):
    return torch.cat([r[key] for r in res], dim=0)


class IDRRetrievalEvaluator(Evaluator):
    """Evaluator for retrieval tasks."""

    retrieval_metrics = MetricCollection(
        {
            "RetrievalMRR": RetrievalMRR(),
            "RetrievalHitRate_top_01": RetrievalHitRate(top_k=1),
            "RetrievalHitRate_top_03": RetrievalHitRate(top_k=3),
            "RetrievalHitRate_top_05": RetrievalHitRate(top_k=5),
            "RetrievalHitRate_top_10": RetrievalHitRate(top_k=10),
            "RetrievalFallOut_top_01": RetrievalFallOut(top_k=1),
            "RetrievalFallOut_top_05": RetrievalFallOut(top_k=5),
            "RetrievalMAP_top_01": RetrievalMAP(top_k=1),
            "RetrievalMAP_top_05": RetrievalMAP(top_k=5),
            "RetrievalPrecision_top_01": RetrievalPrecision(top_k=1),
            "RetrievalPrecision_top_03": RetrievalPrecision(top_k=3),
            "RetrievalPrecision_top_05": RetrievalPrecision(top_k=5),
            "RetrievalPrecision_top_10": RetrievalPrecision(top_k=10),
            "RetrievalRecall_top_01": RetrievalRecall(top_k=1),
            "RetrievalRecall_top_03": RetrievalRecall(top_k=3),
            "RetrievalRecall_top_05": RetrievalRecall(top_k=5),
            "RetrievalRecall_top_10": RetrievalRecall(top_k=10),
            "RetrievalNormalizedDCG": RetrievalNormalizedDCG(),
        }
    )

    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        distance_metric: Optional[Callable] = None,
        name: Optional[str] = None,
        visualize_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer

        if distance_metric is None:
            self.distance_metric = pairwise_cosine_similarity
        else:
            self.distance_metric = distance_metric

        self.name = name or self.model.__class__.__name__
        self.prefix = f"({self.name}) " if self.name else ""

        self.visualize_kwargs = visualize_kwargs or {}

    def finetune(self):
        # No finetuning for retrieval tasks
        pass

    def visualize(self, outs, **kwargs):
        pass

    def retrieval(self, image_embeddings: torch.Tensor, compound_embeddings: torch.Tensor, activities: torch.Tensor):
        dist = self.distance_metric(compound_embeddings, image_embeddings)  # I x C
        indexes = torch.arange(dist.shape[1]).expand(dist.shape)  # I x C
        target = activities.expand(dist.T.shape).T  # I x C

        gene_metrics = self.retrieval_metrics(preds=dist, target=target, indexes=indexes)

        return gene_metrics

    def run(self):
        print("Preparing the data...")
        self.datamodule.prepare_data()
        self.datamodule.setup(stage="predict")
        self.model.eval()

        print("Getting the embeddings for the compounds and images...")
        predict_loaders = self.datamodule.predict_dataloader()

        out_metrics = defaultdict(lambda: defaultdict(list))
        for gene in predict_loaders:
            mol_loader = predict_loaders[gene]["molecule"]
            img_loader = predict_loaders[gene]["image"]
            # List of the embeddings of each group -> Ng x 120 x E
            print("Getting the embeddings for the compounds...")
            compound_embs = self.trainer.predict(self.model, mol_loader)

            print("Getting the embeddings for the images...")
            image_emb = self.trainer.predict(self.model, img_loader)
            # Only needed if more images than batch size (rare) -> Ni x E
            image_emb = concat_from_list_of_dict(image_emb, "image")

            print("Calculating the metrics...")
            for group in compound_embs:
                activities = group["activity_flag"]
                compound_emb = group["compound"]

                gene_group_metrics = self.retrieval(
                    image_embeddings=image_emb, compound_embeddings=compound_emb, activities=activities
                )

                for metric in gene_group_metrics:
                    out_metrics[gene][metric].append(gene_group_metrics[metric])

        print("Aggregating the metrics...")
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

        print("Logging the metrics...")
        for logger in self.trainer.loggers:
            logger.log_metrics(aggregate_metrics)
            logger.save()

        print("Done!")
        return aggregate_metrics
