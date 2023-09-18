import json
import os.path as osp
import pickle
from pathlib import Path
from typing import Optional

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
        test_size: float = 0.2,
        plot: bool = True,
        logistic: bool = True,
        knn: bool = True,
        batch_split: bool = True,
        plate_split: bool = True,
        source_split: bool = True,
        well_split: bool = True,
        fully_random_split: bool = True,
        dmso_normalize: bool = True,
        out_dir: Optional[str] = None,
        name: Optional[str] = None,
        visualize_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer

        self.embeddings_df: Optional[pd.DataFrame] = None

        self.dmso_normalize = dmso_normalize
        self.dmso_embeddings: Optional[dict] = None

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
        if self.out_dir is not None:
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
        if self.dmso_embeddings is not None:
            print("Found dmso_embeddings, skipping getting embeddings...")
            return

        embedding_path = f"{self.out_dir}/dmso_embeddings.pickle" if self.out_dir is not None else None
        if embedding_path and Path(embedding_path).exists():
            print(f'Found dmso_df at "{embedding_path}", skipping getting embeddings...')
            with open(embedding_path, "rb") as f:
                self.dmso_embeddings = pickle.load(f)
            return

        self.datamodule.prepare_data()
        self.datamodule.setup("predict")
        predictions = self.trainer.predict(self.model, self.datamodule.dmso_dataloader())
        keys = list(predictions[0].keys())

        dmso_embeddings_df = pd.DataFrame({key: concat_from_list_of_dict(predictions, key) for key in keys})

        all_batches = dmso_embeddings_df["batch"].unique()

        transform_dict = {}
        for batchi in all_batches:
            dmso_embeddings_batch = np.array(dmso_embeddings_df.query("batch==@batchi").embeddings.to_list())
            spherizer = ZCA_corr()
            spherizer.fit(dmso_embeddings_batch)
            transform_dict[batchi] = spherizer

        self.dmso_embeddings = transform_dict

        if embedding_path:
            with open(embedding_path, "wb") as f:
                pickle.dump(transform_dict, f)

    def get_embeddings(self):
        if self.dmso_normalize:
            self.get_dmso_embeddings()

        embedding_path = f"{self.out_dir}/embeddings.parquet" if self.out_dir is not None else None

        if self.embeddings_df is not None:
            print("Found embeddings_df, skipping getting embeddings...")

        elif embedding_path and Path(embedding_path).exists():
            print(f'Found embeddings at "{embedding_path}", skipping getting embeddings...')
            self.embeddings_df = pd.read_parquet(embedding_path)
        else:
            predictions = self.trainer.predict(self.model, self.datamodule)
            keys = list(predictions[0].keys())

            self.embeddings_df = pd.DataFrame({key: concat_from_list_of_dict(predictions, key) for key in keys})

            self.n_labels = self.embeddings_df["label"].nunique()
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.embeddings_df["label"])

            if embedding_path:
                self.embeddings_df.to_parquet(embedding_path, index=False)

        if self.dmso_normalize and self.dmso_embeddings:
            for batchi in self.embeddings_df["batch"].unique():
                try:
                    dmso_embeddings_batch = np.array(self.embeddings_df.query("batch==@batchi").embeddings.to_list())
                    spherizer = self.dmso_embeddings[batchi]
                    self.embeddings_df.loc[self.embeddings_df.batch == batchi, "embedding"] = spherizer.transform(
                        dmso_embeddings_batch
                    ).tolist()
                except Exception as e:
                    print(f"Error while normalizing batch {batchi}: {e}")

        self.n_labels = self.embeddings_df["label"].nunique()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.embeddings_df["label"])

    def plot_tsne(self, embeddings, col, title="t-SNE"):
        try:
            fig, ax = plt.subplots(figsize=(14, 14))
            sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=self.embeddings_df[col], ax=ax)
            ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            fig.suptitle(title)

            if self.out_dir:
                print(f"Saved {title} to {self.out_dir}/{title.replace(' ', '_')}.png")
                fig.savefig(osp.join(self.out_dir, f"{title.replace(' ', '_')}.png"))
            return fig

        except Exception as e:
            print(f"Error while plotting t-SNE: {e}")

    def plot_embeddings(self, key="batch_effect/Embeddings"):
        print("Fiting t-SNE...")
        tsne = TSNE(n_components=2, random_state=0)
        embeddings = tsne.fit_transform(np.array(self.embeddings_df["embedding"].tolist()))

        images = [
            self.plot_tsne(embeddings, "label", "t-SNE colored by labels"),
            self.plot_tsne(embeddings, "batch", "t-SNE colored by batch"),
            self.plot_tsne(embeddings, "plate", "t-SNE colored by plate"),
            self.plot_tsne(embeddings, "well", "t-SNE colored by well"),
            self.plot_tsne(embeddings, "source", "t-SNE colored by source"),
        ]

        # Log plots to WandB
        try:
            if self.logger:
                # wandb_images = [wandb.Image(image) for image in images]
                self.logger.log_image(key=key, images=images)
                self.logger.save()
        except Exception as e:
            print(f"Error while logging t-SNE plots: {e}")

    def get_metric_dict(self, cls, X_test, y_test, key):
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
            print(f"Saved metrics to {self.out_dir}/{key}_metrics.json")
            with open(f"{self.out_dir}/{key}_metrics.json", "w") as f:
                json.dump(metric_dict, f)

        return metric_dict

    def plot_conf_matrix(self, cls, X_test, y_test, key):
        try:
            labels = self.label_encoder.classes_
            fig, ax = plt.subplots(figsize=(14, 14))
            plot_confusion_matrix(
                self.label_encoder.inverse_transform(y_test),
                self.label_encoder.inverse_transform(cls.predict(X_test)),
                labels=labels,
                ax=ax,
                x_tick_rotation=90,
                title=(title := f"{key}/Confusion matrix"),
            )
            fig.savefig(f"{self.out_dir}/{title.replace(' ', '_')}.png")
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
            fig.savefig(f"{self.out_dir}/{title.replace(' ', '_')}.png")
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
            fig.savefig(f"{self.out_dir}/{title.replace(' ', '_')}.png")
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
            fig.savefig(f"{self.out_dir}/{title.replace(' ', '_')}.png")
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
            fig.savefig(f"{self.out_dir}/{title.replace(' ', '_')}.png")
            return fig
        except Exception as e:
            print(f"Error while plotting precision-recall curves: {e}")

    def plot_metrics(self, cls, X_test, y_test, key):
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

        return images

    def plot_results(self, cls, X_test, y_test, key):
        metric_dict = self.get_metric_dict(cls, X_test, y_test, key)
        metric_images = self.plot_metrics(cls, X_test, y_test, key)

        # Log to WandB
        try:
            if self.logger:
                self.logger.log_metrics(metric_dict)
                self.logger.log_image(key=key, images=metric_images)
                self.logger.save()
        except Exception as e:
            print(f"Error while logging metrics: {e}")

    def not_same_well_cls(self, cls, key="batch_effect/NotSameWell"):
        wells = self.embeddings_df["well"].unique()
        wells = np.random.permutation(wells)

        train_wells = wells[: int(len(wells) * self.train_ratio)]
        test_wells = wells[int(len(wells) * self.train_ratio) :]

        train_df = self.embeddings_df[self.embeddings_df["well"].isin(train_wells)]
        test_df = self.embeddings_df[self.embeddings_df["well"].isin(test_wells)]

        X_train = np.array(train_df["embedding"].tolist())
        y_train = self.label_encoder.transform(train_df["label"].tolist())
        X_test = np.array(test_df["embedding"].tolist())
        y_test = self.label_encoder.transform(test_df["label"].tolist())

        # KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        cls.fit(X_train, y_train)

        self.plot_results(cls, X_test, y_test, key)

    def not_same_batch(self, cls, key="batch_effect/NotSameBatch"):
        batches = self.embeddings_df["batch"].unique()
        batches = np.random.permutation(batches)

        train_batches = batches[: int(len(batches) * self.train_ratio)]
        test_batches = batches[int(len(batches) * self.train_ratio) :]

        train_df = self.embeddings_df[self.embeddings_df["batch"].isin(train_batches)]
        test_df = self.embeddings_df[self.embeddings_df["batch"].isin(test_batches)]

        X_train = np.array(train_df["embedding"].tolist())
        y_train = self.label_encoder.transform(train_df["label"].tolist())
        X_test = np.array(test_df["embedding"].tolist())
        y_test = self.label_encoder.transform(test_df["label"].tolist())

        # KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        cls.fit(X_train, y_train)

        self.plot_results(cls, X_test, y_test, key)

    def not_same_plate(self, cls, key="batch_effect/NotSamePlate"):
        plates = self.embeddings_df["plate"].unique()
        plates = np.random.permutation(plates)

        train_plates = plates[: int(len(plates) * self.train_ratio)]
        test_plates = plates[int(len(plates) * self.train_ratio) :]

        train_df = self.embeddings_df[self.embeddings_df["plate"].isin(train_plates)]
        test_df = self.embeddings_df[self.embeddings_df["plate"].isin(test_plates)]

        X_train = np.array(train_df["embedding"].tolist())
        y_train = self.label_encoder.transform(train_df["label"].tolist())
        X_test = np.array(test_df["embedding"].tolist())
        y_test = self.label_encoder.transform(test_df["label"].tolist())

        # KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        cls.fit(X_train, y_train)

        self.plot_results(cls, X_test, y_test, key)

    def not_same_source(self, cls, key="batch_effect/NotSameSource"):
        sources = self.embeddings_df["source"].unique()
        sources = np.random.permutation(sources)

        train_sources = sources[: int(len(sources) * self.train_ratio)]
        test_sources = sources[int(len(sources) * self.train_ratio) :]

        train_df = self.embeddings_df[self.embeddings_df["source"].isin(train_sources)]
        test_df = self.embeddings_df[self.embeddings_df["source"].isin(test_sources)]

        X_train = np.array(train_df["embedding"].tolist())
        y_train = self.label_encoder.transform(train_df["label"].tolist())
        X_test = np.array(test_df["embedding"].tolist())
        y_test = self.label_encoder.transform(test_df["label"].tolist())

        # KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        cls.fit(X_train, y_train)

        self.plot_results(cls, X_test, y_test, key)

    def fully_random(self, cls, key="batch_effect/FullyRandom"):
        idx = np.random.permutation(len(self.embeddings_df))

        train_idx = idx[: int(len(idx) * self.train_ratio)]
        test_idx = idx[int(len(idx) * self.train_ratio) :]

        train_df = self.embeddings_df.iloc[train_idx]
        test_df = self.embeddings_df.iloc[test_idx]

        X_train = np.array(train_df["embedding"].tolist())
        y_train = self.label_encoder.transform(train_df["label"].tolist())
        X_test = np.array(test_df["embedding"].tolist())
        y_test = self.label_encoder.transform(test_df["label"].tolist())

        # KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
        cls.fit(X_train, y_train)

        return cls, X_test, y_test

        # self.plot_results(cls, X_test, y_test, key)

    def run(self):
        print("Evaluating batch effect")
        print("Evaluating batch effect")
        print("Getting the embeddings...")
        self.get_embeddings()

        if self.plot:
            print("Plotting the embeddings...")
            self.plot_embeddings()

        key_prefix = "batch_effect/regular" if not self.dmso_normalize else "batch_effect/dmso_normalized"

        if self.logistic:
            print("Running Logistic Regressions...")
            cls = LogisticRegression(max_iter=1000)

            if self.fully_random_split:
                print("Fully random split")
                self.fully_random(cls, key=f"{key_prefix}/LR/FullyRandom")
            if self.well_split:
                print("Not same well")
                self.not_same_well_cls(cls, key=f"{key_prefix}/LR/NotSameWell")
            if self.batch_split:
                print("Not same batch")
                self.not_same_batch(cls, key=f"{key_prefix}/LR/NotSameBatch")
            if self.plate_split:
                print("Not same plate")
                self.not_same_plate(cls, key=f"{key_prefix}/LR/NotSamePlate")
            if self.source_split:
                print("Not same source")
                self.not_same_source(cls, key=f"{key_prefix}/LR/NotSameSource")

        if self.knn:
            print("Running KNN Classifier...")
            cls = KNeighborsClassifier(n_neighbors=3, metric="cosine")

            if self.fully_random_split:
                print("Fully random split")
                self.fully_random(cls, key=f"{key_prefix}/KNN/FullyRandom")
            if self.well_split:
                print("Not same well")
                self.not_same_well_cls(cls, key=f"{key_prefix}/KNN/NotSameWell")
            if self.batch_split:
                print("Not same batch")
                self.not_same_batch(cls, key=f"{key_prefix}/KNN/NotSameBatch")
            if self.plate_split:
                print("Not same plate")
                self.not_same_plate(cls, key=f"{key_prefix}/KNN/NotSamePlate")
            if self.source_split:
                print("Not same source")
                self.not_same_source(cls, key=f"{key_prefix}/KNN/NotSameSource")
