import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
from lightning import LightningDataModule, LightningModule, Trainer
from sklearn.decomposition import PCA
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
from src.utils import color_log

py_logger = color_log.get_pylogger(__name__)


def concat_from_list_of_dict_to_list(res, key):
    out = np.concatenate([r[key] for r in res])
    if out.ndim == 2:
        return out.tolist()
    else:
        return out


def concat_from_list_of_dict_to_tensor(res, key):
    if isinstance(res[0][key], torch.Tensor):
        out = torch.cat([r[key] for r in res], dim=0)
    elif isinstance(res[0][key], (int, float)):
        out = [r[key] for r in res]
    else:
        out = concat_from_list_of_dict_to_list(res, key)
    return out


class SimpleRetrievalEvaluator(Evaluator):
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
            "RetrievalPrecision_top_50": RetrievalPrecision(top_k=50),
            "RetrievalRecall_top_01": RetrievalRecall(top_k=1),
            "RetrievalRecall_top_03": RetrievalRecall(top_k=3),
            "RetrievalRecall_top_05": RetrievalRecall(top_k=5),
            "RetrievalRecall_top_10": RetrievalRecall(top_k=10),
            "RetrievalRecall_top_50": RetrievalRecall(top_k=50),
            "RetrievalNormalizedDCG": RetrievalNormalizedDCG(),
        }
    )

    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        distance_metric: Optional[Callable] = None,
        num_to_plot: int = 100,
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

        self.num_to_plot = num_to_plot
        self.metric_keys = list(self.retrieval_metrics.keys())

        self.name = name or self.model.__class__.__name__
        self.prefix = f"({self.name}) " if self.name else ""

        self.visualize_kwargs = visualize_kwargs or {"figsize": (14, 14)}

    def run(self):
        py_logger.info(f"Running {self.name}...")
        py_logger.info("Getting embeddings...")
        predictions = self.trainer.predict(self.model, self.datamodule)

        try:
            self.retrieval_1_to_100(predictions, log=True, save=True)
        except Exception as e:
            py_logger.warning(f"Could not compute 1:100 retrieval metrics: {e}")

        try:
            self.retrieval_1_to_1000(predictions, log=True, save=True)
        except Exception as e:
            py_logger.warning(f"Could not compute 1:1000 retrieval metrics: {e}")

        try:
            self.visualize(predictions, log=True, save=True)
        except Exception as e:
            py_logger.warning(f"Could not visualize: {e}")

    def retrieval_1_to_100(self, predictions, log=True, save=True):
        result_dict = defaultdict(list)

        for batch_res in predictions:
            # batch_res contains image_emb, compound_emb, compound_str, image_id
            image_emb = batch_res["image_emb"]
            compound_emb = batch_res["compound_emb"]

            dist = self.distance_metric(
                image_emb, compound_emb
            )  # Similarity matrix between images and compounds: 100 x 100

            indexes_mol_to_img = torch.arange(dist.shape[1]).expand(
                dist.shape
            )  # 100 x 100 matrix with the indexes of the compounds
            indexes_img_to_mol = indexes_mol_to_img.transpose(0, 1)  # 100 x 100 matrix with the indexes of the images
            target = torch.eye(dist.shape[1])  # Identity matrix: 100 x 100

            res_mol_to_img = self.retrieval_metrics(
                preds=dist, target=target, indexes=indexes_mol_to_img
            )  # Dictionary with the metrics for mol to img
            res_img_to_mol = self.retrieval_metrics(
                preds=dist, target=target, indexes=indexes_img_to_mol
            )  # Dictionary with the metrics for img to mol

            for metric in self.metric_keys:
                result_dict[f"retrieval/1:100/mol_to_img/{metric}"].append(res_mol_to_img[metric])
                result_dict[f"retrieval/1:100/img_to_mol/{metric}"].append(res_img_to_mol[metric])
                result_dict[f"retrieval/1:100/avg/{metric}"].append(
                    (res_mol_to_img[metric] + res_img_to_mol[metric]) / 2
                )

        for metric in self.metric_keys:
            result_dict[f"retrieval/1:100/mol_to_img/{metric}_avg"] = np.mean(
                result_dict[f"retrieval/1:100/mol_to_img/{metric}"]
            )
            result_dict[f"retrieval/1:100/img_to_mol/{metric}_avg"] = np.mean(
                result_dict[f"retrieval/1:100/img_to_mol/{metric}"]
            )
            result_dict[f"retrieval/1:100/avg/{metric}_avg"] = np.mean(result_dict[f"retrieval/1:100/avg/{metric}"])
            result_dict[f"retrieval/1:100/mol_to_img/{metric}_std"] = np.std(
                result_dict[f"retrieval/1:100/mol_to_img/{metric}"]
            )
            result_dict[f"retrieval/1:100/img_to_mol/{metric}_std"] = np.std(
                result_dict[f"retrieval/1:100/img_to_mol/{metric}"]
            )
            result_dict[f"retrieval/1:100/avg/{metric}_std"] = np.std(result_dict[f"retrieval/1:100/avg/{metric}"])

        if log:
            self.log_metrics(result_dict)

        if save:
            self.save_metrics(result_dict, name="metrics_1_to_100.json")

        return result_dict

    def retrieval_1_to_1000(self, predictions, log=True, save=True):
        result_dict = defaultdict(list)
        keys = predictions[0].keys()

        n = len(predictions)
        for i in range(0, n, 10):
            batch_res = {k: concat_from_list_of_dict_to_tensor(predictions[i : i + 10], k) for k in keys}

            image_emb = batch_res["image_emb"]
            compound_emb = batch_res["compound_emb"]

            dist = self.distance_metric(
                image_emb, compound_emb
            )  # Similarity matrix between images and compounds: 100 x 100

            indexes_mol_to_img = torch.arange(dist.shape[1]).expand(
                dist.shape
            )  # 100 x 100 matrix with the indexes of the compounds
            indexes_img_to_mol = indexes_mol_to_img.transpose(0, 1)  # 100 x 100 matrix with the indexes of the images
            target = torch.eye(dist.shape[1])  # Identity matrix: 100 x 100

            res_mol_to_img = self.retrieval_metrics(
                preds=dist, target=target, indexes=indexes_mol_to_img
            )  # Dictionary with the metrics for mol to img
            res_img_to_mol = self.retrieval_metrics(
                preds=dist, target=target, indexes=indexes_img_to_mol
            )  # Dictionary with the metrics for img to mol

            for metric in self.metric_keys:
                result_dict[f"retrieval/1:1000/mol_to_img/{metric}"].append(res_mol_to_img[metric])
                result_dict[f"retrieval/1:1000/img_to_mol/{metric}"].append(res_img_to_mol[metric])
                result_dict[f"retrieval/1:1000/avg/{metric}"].append(
                    (res_mol_to_img[metric] + res_img_to_mol[metric]) / 2
                )

        for metric in self.metric_keys:
            result_dict[f"retrieval/1:1000/mol_to_img/{metric}_avg"] = np.mean(
                result_dict[f"retrieval/1:1000/mol_to_img/{metric}"]
            )
            result_dict[f"retrieval/1:1000/img_to_mol/{metric}_avg"] = np.mean(
                result_dict[f"retrieval/1:1000/img_to_mol/{metric}"]
            )
            result_dict[f"retrieval/1:1000/avg/{metric}_avg"] = np.mean(result_dict[f"retrieval/1:1000/avg/{metric}"])
            result_dict[f"retrieval/1:1000/mol_to_img/{metric}_std"] = np.std(
                result_dict[f"retrieval/1:1000/mol_to_img/{metric}"]
            )
            result_dict[f"retrieval/1:1000/img_to_mol/{metric}_std"] = np.std(
                result_dict[f"retrieval/1:1000/img_to_mol/{metric}"]
            )
            result_dict[f"retrieval/1:1000/avg/{metric}_std"] = np.std(result_dict[f"retrieval/1:1000/avg/{metric}"])

        if log:
            self.log_metrics(result_dict)

        if save:
            self.save_metrics(result_dict, name="metrics_1_to_1000.json")

        return result_dict

    def visualize(self, predictions, log=True, save=True):
        keys = predictions[0].keys()
        res = {k: concat_from_list_of_dict_to_list(predictions[: self.num_to_plot], k) for k in keys}

        sim = F.cosine_similarity(res["image_emb"], res["compound_emb"], dim=1)
        total_sim = torch.cat([sim, sim], dim=0).cpu().numpy()
        total_emb = torch.cat([res["image_emb"], res["compound_emb"]], dim=0).cpu().numpy()
        labels = res["image_id"] + res["compound_str"]
        sources = [x.split("__")[0] for x in res["image_id"]] + ["compound"] * len(res["compound_emb"])

        pca = PCA(n_components=2)
        proj = pca.fit_transform(total_emb)

        proj_dict = {
            "x": proj[:, 0],
            "y": proj[:, 1],
            "type": ["image"] * len(res["image_emb"]) + ["compound"] * len(res["compound_emb"]),
            "source": sources,
            "labels": labels,
            "pair_id": list(range(len(res["image_emb"]))) + list(range(len(res["compound_emb"]))),
            "sim": total_sim,
        }

        proj_df = pd.DataFrame(proj_dict)

        fig = px.scatter(
            proj_df,
            x="x",
            y="y",
            color="sources",
            size="sim",
            symbol="type",
            hover_data=["labels", "pair_id", "sim"],
            **self.visualize_kwargs,
        )

        if save:
            self.save_visualization(fig, name="visualization.html")

        if log:
            self.log_visualization(fig)

    def log_metrics(self, metrics):
        try:
            for logger in self.trainer.loggers:
                logger.log_metrics(metrics)
                logger.save()
        except Exception as e:
            py_logger.warning(f"Could not log metrics: {e}")

    def save_metrics(self, metrics, name="metrics.json"):
        try:
            out = Path(self.trainer.default_root_dir) / name
            if not out.parent.exists():
                print("Creating parent directory")
                out.parent.mkdir(parents=True)
            with open(out, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            py_logger.warning(f"Could not save metrics: {e}")

    def save_visualization(self, fig, name="visualization.html"):
        try:
            fig.write_html(Path(self.trainer.default_root_dir) / name)
        except Exception as e:
            py_logger.warning(f"Could not save visualization: {e}")

    def log_visualization(self, fig):
        for logger in self.trainer.loggers:
            try:
                logger.experiment.log({"visualization": fig})
                logger.save()
            except Exception as e:
                py_logger.warning(f"Could not log visualization to {logger}: {e}")
