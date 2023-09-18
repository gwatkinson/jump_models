import io
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
from src.utils import pylogger

py_logger = pylogger.get_pylogger(__name__)


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
        embedding_path: Optional[str] = None,
        name: Optional[str] = None,
        visualize_kwargs: Optional[dict] = None,
    ):
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer

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

        self.scorers = {
            "accuracy": make_scorer(accuracy_score),
            "balanced_accuracy": make_scorer(balanced_accuracy_score),
            "prec": make_scorer(precision_score, average="macro"),
            "recall": make_scorer(recall_score, average="macro"),
            "f1": make_scorer(f1_score, average="macro"),
            "roc_auc": make_scorer(roc_auc_score, average="macro"),
            "top_3": make_scorer(top_k_accuracy_score, k=3),
            "top_5": make_scorer(top_k_accuracy_score, k=5),
            "top_10": make_scorer(top_k_accuracy_score, k=10),
        }

        self.embedding_path = embedding_path
        self.name = name or self.model.__class__.__name__
        self.prefix = f"({self.name}) " if self.name else ""

        self.visualize_kwargs = visualize_kwargs or {}

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

    def get_embeddings(self):
        predictions = self.trainer.predict(self.model, self.datamodule)
        keys = list(predictions[0].keys())

        self.embeddings_df = pd.DataFrame({key: concat_from_list_of_dict(predictions, key) for key in keys})

        self.n_labels = self.embeddings_df["label"].nunique()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.embeddings_df["label"])

        if self.embedding_path:
            self.embeddings_df.to_parquet(self.embedding_path, index=False)

    def plot_results(self, y_test, y_predict, y_probas, key):
        # Metrics
        metric_dict = {k: scorer(y_test, y_predict) for k, scorer in self.scorers.items()}

        images = []

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(14, 14))
        labels = self.label_encoder.classes_
        plot_confusion_matrix(y_test, y_predict, labels=labels, ax=ax, x_tick_rotation=90)
        conf_buf = io.BytesIO()
        fig.savefig(conf_buf)
        plt.close(fig)
        images.append(conf_buf)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_roc(
            y_test,
            y_probas,
            ax=ax,
            plot_macro=True,
            plot_micro=False,
            classes_to_plot=[],
            title="Macro-average ROC curve",
        )
        roc_buf = io.BytesIO()
        fig.savefig(roc_buf)
        plt.close(fig)
        images.append(roc_buf)

        # Plot total ROC curve
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_roc(y_test, y_probas, ax=ax, plot_macro=True, plot_micro=True, classes_to_plot=None, title="ROC curves")
        tot_roc_buf = io.BytesIO()
        fig.savefig(tot_roc_buf)
        plt.close(fig)
        images.append(tot_roc_buf)

        # Plot precision-recall curve
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_precision_recall(
            y_test, y_probas, ax=ax, plot_micro=True, classes_to_plot=[], title="Micro-average precision-recall curve"
        )
        pr_buf = io.BytesIO()
        fig.savefig(pr_buf)
        plt.close(fig)
        images.append(pr_buf)

        # Plot total precision-recall curve
        fig, ax = plt.subplots(figsize=(14, 14))
        plot_precision_recall(
            y_test, y_probas, ax=ax, plot_micro=True, classes_to_plot=None, title="Precision-recall curves"
        )
        tot_pr_buf = io.BytesIO()
        fig.savefig(tot_pr_buf)
        plt.close(fig)
        images.append(tot_pr_buf)

        # Log to WandB
        self.logger.log_metrics(metric_dict)
        self.logger.log_image(key=key, images=images)
        self.logger.save()

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

        y_predict = cls.predict(X_test)
        y_probas = cls.predict_proba(X_test)

        self.plot_results(y_test, y_predict, y_probas, key)

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

        y_predict = cls.predict(X_test)
        y_probas = cls.predict_proba(X_test)

        self.plot_results(y_test, y_predict, y_probas, key)

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

        y_predict = cls.predict(X_test)
        y_probas = cls.predict_proba(X_test)

        self.plot_results(y_test, y_predict, y_probas, key)

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

        y_predict = cls.predict(X_test)
        y_probas = cls.predict_proba(X_test)

        self.plot_results(y_test, y_predict, y_probas, key)

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

        y_predict = cls.predict(X_test)
        y_probas = cls.predict_proba(X_test)

        self.plot_results(y_test, y_predict, y_probas, key)

    def plot_tsne(self, embeddings, col, title="t-SNE"):
        fig, ax = plt.subplots(figsize=(14, 14))
        sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=self.embeddings_df[col], ax=ax)
        fig.suptitle(title)
        emb_buf = io.BytesIO()
        fig.savefig(emb_buf)
        plt.close(fig)

        return emb_buf

    def plot_embeddings(self, key="batch_effect/Embeddings"):
        tsne = TSNE(n_components=2, random_state=0)
        embeddings = tsne.fit_transform(np.array(self.embeddings_df["embedding"].tolist()))

        images = []

        # Plot with regards to label
        images.append(self.plot_tsne(embeddings, "label", "t-SNE colored by labels"))

        # Plot with regards to batch
        images.append(self.plot_tsne(embeddings, "batch", "t-SNE colored by batch"))

        # Plot with regards to plate
        images.append(self.plot_tsne(embeddings, "plate", "t-SNE colored by plate"))

        # Plot with regards to well
        images.append(self.plot_tsne(embeddings, "well", "t-SNE colored by well"))

        # Plot with regards to source
        images.append(self.plot_tsne(embeddings, "source", "t-SNE colored by source"))

        # Log plots to WandB
        self.logger.log_image(key=key, images=images)
        self.logger.save()

    def run(self):
        py_logger.info("=== Evaluating batch effect")
        py_logger.info("Getting the embeddings...")
        self.get_embeddings()

        if self.plot:
            py_logger.info("Plotting the embeddings...")
            self.plot_embeddings()

        if self.logistic:
            py_logger.info("Running Logistic Regressions...")
            cls = LogisticRegression(max_iter=1000)

            if self.fully_random_split:
                py_logger.info("Fully random split")
                self.fully_random(cls)
            if self.well_split:
                py_logger.info("Not same well")
                self.not_same_well_cls(cls)
            if self.batch_split:
                py_logger.info("Not same batch")
                self.not_same_batch(cls)
            if self.plate_split:
                py_logger.info("Not same plate")
                self.not_same_plate(cls)
            if self.source_split:
                py_logger.info("Not same source")
                self.not_same_source(cls)

        if self.knn:
            py_logger.info("Running KNN Classifier...")
            cls = KNeighborsClassifier(n_neighbors=3, metric="cosine")

            if self.fully_random_split:
                py_logger.info("Fully random split")
                self.fully_random(cls)
            if self.well_split:
                py_logger.info("Not same well")
                self.not_same_well_cls(cls)
            if self.batch_split:
                py_logger.info("Not same batch")
                self.not_same_batch(cls)
            if self.plate_split:
                py_logger.info("Not same plate")
                self.not_same_plate(cls)
            if self.source_split:
                py_logger.info("Not same source")
                self.not_same_source(cls)
