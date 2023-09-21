import json
import os.path as osp
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers.wandb import WandbLogger
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall, plot_roc
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from src.eval import Evaluator
from src.eval.batch_effect.spherize import ZCA_corr

# from src.utils import pylogger

# print pylogger.get_pylogger(__name__)


def concat_from_list_of_dict(res, key):
    out = np.concatenate([r[key] for r in res])
    if out.ndim == 2:
        return out.tolist()
    else:
        return out


class BatchEffectEvaluator(Evaluator):
    """Evaluator for retrieval tasks.

    This gets the embeddings, then uses metadata and standard ML models
    to evaluate the performance. This also visualizes the embeddings
    using t-SNE, colored by metadata.
    """

    def __init__(
        self,
        model: LightningModule,
        datamodule: LightningDataModule,
        trainer: Trainer,
        embedding_col: str = "embedding",
        test_size: float = 0.2,
        plot: bool = True,
        logistic: bool = True,
        knn: bool = True,
        batch_split: bool = True,
        plate_split: bool = True,
        source_split: bool = True,
        well_split: bool = True,
        fully_random_split: bool = True,
        nruns: int = 5,
        dmso_normalize: Union[str, bool] = "batch",
        normalize_cls=ZCA_corr,
        out_dir: Optional[str] = None,
        name: Optional[str] = None,
        visualize_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer

        self.embedding_col = embedding_col

        self.embeddings_df: Optional[pd.DataFrame] = None

        self.dmso_normalize: str | bool = dmso_normalize
        self.dmso_transforms: Optional[dict] = None
        self.normalize_cls = normalize_cls
        self.normalize_cls_str = normalize_cls.__name__ if normalize_cls else "None"

        self.nruns = nruns if nruns > 1 else 0

        self.plot = plot
        self.logistic = logistic
        self.knn = knn

        self.batch_split = batch_split
        self.plate_split = plate_split
        self.source_split = source_split
        self.well_split = well_split
        self.fully_random_split = fully_random_split

        self.test_size = test_size
        self.train_ratio = 1 - self.test_size

        self.out_dir = out_dir
        if self.out_dir is not None and not Path(self.out_dir).exists():
            print(f"Creating output directory {self.out_dir}...")
            Path(self.out_dir).mkdir(parents=True, exist_ok=True)

        self.name = name or self.model.__class__.__name__
        self.prefix = f"({self.name}) " if self.name else ""

        self.visualize_kwargs = visualize_kwargs or {}

        self.logger = None
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                self.logger = logger
                break

    def finetune(self):
        # No finetuning for retrieval tasks
        pass

    def evaluate(self):
        pass

    def visualize(self, outs, **kwargs):
        pass

    def get_dmso_embeddings(self):
        if self.dmso_transforms is not None:
            print("Found dmso_embeddings, skipping getting embeddings...")
            return

        embedding_path = (
            osp.join(self.out_dir, f"dmso_{self.dmso_normalize}_transforms_{self.normalize_cls_str}.pickle")
            if self.out_dir is not None
            else None
        )
        if embedding_path and Path(embedding_path).exists():
            print(f'Found dmso_df at "{embedding_path}", skipping getting embeddings...')
            with open(embedding_path, "rb") as f:
                self.dmso_transforms = pickle.load(f)
            return

        dmso_embeddings_df_path = (
            osp.join(self.out_dir, f"dmso_{self.dmso_normalize}_embeddings_{self.normalize_cls_str}_df.parquet")
            if self.out_dir is not None
            else None
        )
        if dmso_embeddings_df_path and Path(dmso_embeddings_df_path).exists():
            dmso_embeddings_df = pd.read_parquet(dmso_embeddings_df_path)
        else:
            self.datamodule.prepare_data()
            self.datamodule.setup("predict")
            print("Predicting on DMSO...")
            predictions = self.trainer.predict(self.model, self.datamodule.dmso_dataloader())
            keys = list(predictions[0].keys())
            dmso_embeddings_df = pd.DataFrame({key: concat_from_list_of_dict(predictions, key) for key in keys})

            dmso_embeddings_df = dmso_embeddings_df.assign(
                well=lambda x: x["source"] + "__" + x["batch"] + "__" + x["plate"] + "__" + x["well"],
                plate=lambda x: x["source"] + "__" + x["batch"] + "__" + x["plate"],
                batch=lambda x: x["source"] + "__" + x["batch"],
            )

        all_batches = dmso_embeddings_df[self.dmso_normalize].unique()

        print("Fitting the transforms on batches...")
        transform_dict = {}
        for batchi in all_batches:
            dmso_embeddings_batch = np.array(
                dmso_embeddings_df.query(f"{self.dmso_normalize}==@batchi")[self.embedding_col].to_list()
            )
            spherizer = self.normalize_cls()
            spherizer.fit(dmso_embeddings_batch)
            transform_dict[batchi] = spherizer

        self.dmso_transforms = transform_dict

        if embedding_path:
            dmso_embeddings_df.to_parquet(dmso_embeddings_df_path)
            with open(embedding_path, "wb") as f:
                pickle.dump(transform_dict, f)

    def normalize_embeddings(self):
        if self.embeddings_df is None:
            raise ValueError("embeddings_df is None, please run get_embeddings first")

        if self.dmso_normalize and self.dmso_transforms:
            for batchi in self.embeddings_df[self.dmso_normalize].unique():
                try:
                    dmso_embeddings_batch = np.array(
                        self.embeddings_df.query(f"{self.dmso_normalize}==@batchi")[self.embedding_col].to_list()
                    )
                    spherizer = self.dmso_transforms[batchi]
                    self.embeddings_df.loc[
                        self.embeddings_df[self.dmso_normalize] == batchi, "normed_embedding"
                    ] = pd.Series(spherizer.transform(dmso_embeddings_batch).tolist()).values
                except Exception as e:
                    print(f"Error while normalizing batch {batchi}: {e}")
        else:
            self.embeddings_df["normed_embedding"] = self.embeddings_df[self.embedding_col]

    def get_embeddings(self):
        embedding_path = osp.join(self.out_dir, "embeddings.parquet") if self.out_dir is not None else None

        if self.embeddings_df is not None:
            print("Found embeddings_df, skipping getting embeddings...")

        elif embedding_path and Path(embedding_path).exists():
            print(f'Found embeddings at "{embedding_path}", skipping getting embeddings...')
            self.embeddings_df = pd.read_parquet(embedding_path)
        else:
            predictions = self.trainer.predict(self.model, self.datamodule)
            keys = list(predictions[0].keys())

            self.embeddings_df = pd.DataFrame({key: concat_from_list_of_dict(predictions, key) for key in keys})

            self.embeddings_df = self.embeddings_df.assign(
                well=lambda x: x["source"] + "__" + x["batch"] + "__" + x["plate"] + "__" + x["well"],
                plate=lambda x: x["source"] + "__" + x["batch"] + "__" + x["plate"],
                batch=lambda x: x["source"] + "__" + x["batch"],
            )

            self.n_labels = self.embeddings_df["label"].nunique()
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.embeddings_df["label"])

            if embedding_path:
                self.embeddings_df.to_parquet(embedding_path, index=False)

        if self.dmso_normalize:
            self.get_dmso_embeddings()

        self.normalize_embeddings()

        self.n_labels = self.embeddings_df["label"].nunique()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.embeddings_df["label"])

    def plot_tsne(self, col, title="t-SNE", remove_legend=False):
        if self.embeddings_df is None:
            raise ValueError("embeddings_df is None, please run get_embeddings first")

        try:
            fig, ax = plt.subplots(figsize=(14, 14))
            sns.scatterplot(
                data=self.embeddings_df,
                x="x",
                y="y",
                hue=col,
                ax=ax,
            )
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            if remove_legend:
                ax.get_legend().remove()
            fig.suptitle(title, fontsize=18)
            fig.tight_layout()

            if self.out_dir:
                out_path = osp.join(self.out_dir, f"{title.replace(' ', '_')}.png")
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                print(f"Saved {title} to {out_path}")
                fig.savefig(out_path)
            return fig

        except Exception as e:
            print(f"Error while plotting t-SNE: {e}")

    def plot_embeddings(self, key="batch_effect/Embeddings"):
        if self.embeddings_df is None:
            raise ValueError("embeddings_df is None, please run get_embeddings first")

        print("Fitting t-SNE...")
        tsne = TSNE(n_components=2, random_state=0)
        embeddings = tsne.fit_transform(np.array(self.embeddings_df[self.embedding_col].tolist()))

        self.embeddings_df = self.embeddings_df.assign(x=embeddings[:, 0], y=embeddings[:, 1]).sample(frac=1)

        images = [
            self.plot_tsne("label", f"{key}/t-SNE colored by labels"),
            self.plot_tsne("batch", f"{key}/t-SNE colored by batch", remove_legend=True),
            self.plot_tsne("plate", f"{key}/t-SNE colored by plate", remove_legend=True),
            self.plot_tsne("well", f"{key}/t-SNE colored by well", remove_legend=True),
            self.plot_tsne("source", f"{key}/t-SNE colored by source"),
        ]

        # Log plots to WandB
        try:
            if self.logger:
                # wandb_images = [wandb.Image(image) for image in images]
                self.logger.log_image(key=key, images=images)
                self.logger.save()
        except Exception as e:
            print(f"Error while logging t-SNE plots: {e}")

    def get_metric_dict(self, cls, X_test, y_test, key, log=True):
        # Metrics
        labels = np.arange(len(self.label_encoder.classes_))
        self.scorers = {
            "accuracy": make_scorer(accuracy_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "prec": make_scorer(precision_score, average="macro", labels=labels),
            "recall": make_scorer(recall_score, average="macro", labels=labels),
            "f1": make_scorer(f1_score, average="macro", labels=labels),
            "roc_auc": make_scorer(roc_auc_score, average="macro", multi_class="ovr", needs_proba=True, labels=labels),
            "top_3": make_scorer(top_k_accuracy_score, k=3, labels=labels, needs_proba=True),
            "top_5": make_scorer(top_k_accuracy_score, k=5, labels=labels, needs_proba=True),
            "top_10": make_scorer(top_k_accuracy_score, k=10, labels=labels, needs_proba=True),
        }

        metric_dict = {}
        for k, scorer in self.scorers.items():
            try:
                metric_dict[f"{key}/{k}"] = scorer(cls, X_test, y_test)
            except Exception as e:
                print(f"Error while computing {k}: {e}")

        if self.out_dir:
            out_path = f"{self.out_dir}/{key}_metrics.json".replace(" ", "_")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saved metrics to {out_path}")
            with open(out_path, "w") as f:
                json.dump(metric_dict, f)

        if log:
            try:
                if self.logger:
                    self.logger.log_metrics(metric_dict)
                    self.logger.save()
            except Exception as e:
                print(f"Error while logging metrics: {e}")

        return metric_dict

    def plot_conf_matrix(self, cls, X_test, y_test, key):
        try:
            labels = self.label_encoder.classes_
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_confusion_matrix(
                self.label_encoder.inverse_transform(y_test),
                self.label_encoder.inverse_transform(cls.predict(X_test)),
                labels=labels,
                text_fontsize="small",
                hide_counts=True,
                ax=ax,
                x_tick_rotation=90,
                title=(title := f"{key}/Confusion matrix"),
            )
            out_path = f"{self.out_dir}/{title.replace(' ', '_')}.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            return fig
        except Exception as e:
            print(f"Error while plotting confusion matrix: {e}")

    def plot_macro_roc_curve(self, cls, X_test, y_test, key):
        try:
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_roc(
                self.label_encoder.inverse_transform(y_test),
                cls.predict_proba(X_test),
                ax=ax,
                plot_macro=True,
                plot_micro=False,
                classes_to_plot=[],
                title=(title := f"{key}/Macro-average ROC curve"),
            )
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            fig.tight_layout()
            out_path = f"{self.out_dir}/{title.replace(' ', '_')}.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            return fig
        except Exception as e:
            print(f"Error while plotting macro-average ROC curve: {e}")

    def plot_roc_curves(self, cls, X_test, y_test, key):
        try:
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_roc(
                self.label_encoder.inverse_transform(y_test),
                cls.predict_proba(X_test),
                ax=ax,
                plot_macro=True,
                plot_micro=True,
                classes_to_plot=None,
                title=(title := f"{key}/ROC curves"),
            )
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            fig.tight_layout()
            out_path = f"{self.out_dir}/{title.replace(' ', '_')}.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            return fig
        except Exception as e:
            print(f"Error while plotting ROC curves: {e}")

    def plot_micro_pr_curve(self, cls, X_test, y_test, key):
        try:
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_precision_recall(
                self.label_encoder.inverse_transform(y_test),
                cls.predict_proba(X_test),
                ax=ax,
                plot_micro=True,
                classes_to_plot=[],
                title=(title := f"{key}/Micro-average precision-recall curve"),
            )
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            fig.tight_layout()
            out_path = f"{self.out_dir}/{title.replace(' ', '_')}.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            return fig
        except Exception as e:
            print(f"Error while plotting micro-average precision-recall curve: {e}")

    def plot_pr_curves(self, cls, X_test, y_test, key):
        try:
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_precision_recall(
                self.label_encoder.inverse_transform(y_test),
                cls.predict_proba(X_test),
                ax=ax,
                plot_micro=True,
                classes_to_plot=None,
                title=(title := f"{key}/Precision-recall curves"),
            )
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            fig.tight_layout()
            out_path = f"{self.out_dir}/{title.replace(' ', '_')}.png"
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path)
            return fig
        except Exception as e:
            print(f"Error while plotting precision-recall curves: {e}")

    def log_images(self, key, images):
        try:
            if self.logger:
                self.logger.log_image(key=key, images=images)
                self.logger.save()
        except Exception as e:
            print(f"Error while logging metrics: {e}")

    def plot_metrics(self, cls, X_test, y_test, key, log=True):
        images = []

        for func in [
            self.plot_conf_matrix,
            self.plot_macro_roc_curve,
            self.plot_roc_curves,
            self.plot_micro_pr_curve,
            self.plot_pr_curves,
        ]:
            fig = func(cls, X_test, y_test, key)
            if fig:
                images.append(fig)

        # Log to WandB
        if log:
            self.log_images(key, images)

        return images

    def single_run(self, cls, col, key, plot_all=False, log=True):
        try:
            if col == "random":
                idx = np.random.permutation(len(self.embeddings_df))
            else:
                idx = self.embeddings_df[col].unique()
                idx = np.random.permutation(idx)

            idx = idx[: int(len(idx) * self.train_ratio)]
            idx = idx[int(len(idx) * self.train_ratio) :]

            train_df = self.embeddings_df[self.embeddings_df["batch"].isin(idx)]
            test_df = self.embeddings_df[self.embeddings_df["batch"].isin(idx)]

            X_train = np.array(train_df[self.embedding_col].tolist())
            y_train = self.label_encoder.transform(train_df["label"].tolist())
            X_test = np.array(test_df[self.embedding_col].tolist())
            y_test = self.label_encoder.transform(test_df["label"].tolist())

            cls.fit(X_train, y_train)

            self.get_metric_dict(cls, X_test, y_test, key, log=log)  # calculate metrics and log to wandb

            if plot_all:
                self.plot_metrics(cls, X_test, y_test, key, log=log)  # plot metrics and log to wandb
            else:
                fig = self.plot_conf_matrix(cls, X_test, y_test, key)
                if fig:
                    self.log_images(key, [fig])

        except Exception as e:
            print(f"Error while running {key}: {e}")

    def multi_run(self, nruns, cls, col, key, plot_all=False, log=True):
        final_metric_dict = defaultdict(list)

        for _ in range(nruns):
            metric_dict = self.not_same_col_cls(cls, col, key, plot_all=plot_all, log=False)
            for k, v in metric_dict.items():
                final_metric_dict[k].append(v)

        out_dict = {}
        for k, v in final_metric_dict.items():
            out_dict[f"{k}_mean"] = np.mean(v)
            out_dict[f"{k}_std"] = np.std(v)
            out_dict[f"{k}_min"] = np.min(v)
            out_dict[f"{k}_max"] = np.max(v)

        if self.out_dir:
            out_path = f"{self.out_dir}/{key}_metrics.json".replace(" ", "_")
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            print(f"Saved metrics to {out_path}")
            with open(out_path, "w") as f:
                json.dump(out_dict, f)

        if log:
            try:
                if self.logger:
                    self.logger.log_metrics(out_dict)
                    self.logger.save()
            except Exception as e:
                print(f"Error while logging metrics: {e}")

        return out_dict

    def not_same_col_cls(self, cls, col, key, plot_all=False, log=True):
        if self.nruns:
            return self.multi_run(self.nruns, cls, col, key, plot_all=plot_all, log=log)
        else:
            return self.single_run(cls, col, key, plot_all=plot_all, log=log)

    def run(self):
        print("Evaluating batch effect")
        print("Getting the embeddings...")
        self.get_embeddings()

        key_prefix = (
            "batch_effect/regular"
            if not self.dmso_normalize
            else f"batch_effect/dmso_{self.dmso_normalize}_normalized_{self.normalize_cls_str}"
        )

        if self.plot:
            print("Plotting the embeddings...")
            self.plot_embeddings(key=f"{key_prefix}/Embeddings")

        if self.logistic:
            try:
                print("Running Logistic Regressions...")
                cls = LogisticRegression(max_iter=1000)

                if self.fully_random_split:
                    print("Fully random split")
                    self.not_same_col_cls(cls, "random", key=f"{key_prefix}/LR/FullyRandom")
                if self.well_split:
                    print("Not same well")
                    self.not_same_col_cls(cls, "well", key=f"{key_prefix}/LR/NotSameWell")
                if self.batch_split:
                    print("Not same batch")
                    self.not_same_col_cls(cls, "batch", key=f"{key_prefix}/LR/NotSameBatch")
                if self.plate_split:
                    print("Not same plate")
                    self.not_same_col_cls(cls, "plate", key=f"{key_prefix}/LR/NotSamePlate")
                if self.source_split:
                    print("Not same source")
                    self.not_same_col_cls(cls, "source", key=f"{key_prefix}/LR/NotSameSource")
            except Exception as e:
                print(f"Error while running Logistic Regression: {e}")

        if self.knn:
            try:
                print("Running KNN Classifier...")
                cls = KNeighborsClassifier(n_neighbors=3, metric="cosine")

                if self.fully_random_split:
                    print("Fully random split")
                    self.not_same_col_cls(cls, "random", key=f"{key_prefix}/KNN/FullyRandom")
                if self.well_split:
                    print("Not same well")
                    self.not_same_col_cls(cls, "well", key=f"{key_prefix}/KNN/NotSameWell")
                if self.batch_split:
                    print("Not same batch")
                    self.not_same_col_cls(cls, "batch", key=f"{key_prefix}/KNN/NotSameBatch")
                if self.plate_split:
                    print("Not same plate")
                    self.not_same_col_cls(cls, "plate", key=f"{key_prefix}/KNN/NotSamePlate")
                if self.source_split:
                    print("Not same source")
                    self.not_same_col_cls(cls, "source", key=f"{key_prefix}/KNN/NotSameSource")
            except Exception as e:
                print(f"Error while running KNN Classifier: {e}")
