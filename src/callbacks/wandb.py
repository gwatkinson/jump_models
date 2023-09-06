"""Defines callbacks that add logging capabilities to the training process
using the wandb logger."""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

import wandb
from src.utils import pylogger

# from src.utils.visualisation import pp_matrix

py_logger = pylogger.get_pylogger(__name__)


class WandbTrainingCallback(Callback):
    def __init__(
        self,
        watch: bool = True,
        watch_log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        log_freq: int = 100,
        log_graph: bool = True,
    ):
        super().__init__()
        self.watch = watch
        self.log_freq = log_freq
        self.watch_log = watch_log
        self.log_graph = log_graph
        self.no_logger = False

        self.logger = None

    def setup(self, trainer, pl_module, stage=None):
        if self.logger is None:
            loggers = trainer.loggers

            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    self.logger = logger
                    break

            if self.logger is None:
                py_logger.warning("No WandbLogger found. WandbCallback will not log anything.")
                self.no_logger = True

    def on_train_start(self, trainer, pl_module):
        if self.watch and not self.no_logger:
            self.logger.watch(pl_module, log=self.watch_log, log_freq=self.log_freq, log_graph=self.log_graph)

    def on_train_end(self, trainer, pl_module):
        if self.watch and not self.no_logger:
            self.logger.experiment.unwatch(pl_module)


class WandbPlottingCallback(WandbTrainingCallback):
    def __init__(
        self,
        watch: bool = True,
        watch_log: Optional[Literal["gradients", "parameters", "all"]] = "all",
        log_freq: int = 100,
        log_graph: bool = True,
        plot_every_n_epoch: int = 2,
        prefix: Optional[str] = None,
        fig_kws: Optional[dict] = None,
        plot_kws: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(watch=watch, watch_log=watch_log, log_freq=log_freq, log_graph=log_graph)
        self.tables = None
        self.num_figs = None
        self.fig_kws = fig_kws or {}
        self.plot_kws = plot_kws or {}
        self.plot_every_n_epoch = plot_every_n_epoch

        if prefix is None:
            self.prefix = ""
        elif prefix.endswith("/"):
            self.prefix = prefix
        else:
            self.prefix = prefix + "/"

    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage=stage)

        if not self.no_logger:
            if self.tables is None:
                plot_metrics = pl_module.train_plot_metrics
                columns = ["epoch", *[k.replace("train/", "") for k in plot_metrics.keys()]]
                self.tables = {
                    "train": wandb.Table(columns=columns),
                    "val": wandb.Table(columns=columns),
                    "test": wandb.Table(columns=columns),
                }

            if self.num_figs is None:
                plot_metrics = pl_module.train_plot_metrics
                self.num_figs = len(plot_metrics)

            if hasattr(trainer.datamodule, "num_to_labels"):
                self.num_to_labels = trainer.datamodule.num_to_labels
            else:
                self.num_to_labels = None

    def on_epoch_end_plotting(self, trainer, pl_module, phase="train"):
        if not self.no_logger and self.num_figs > 0:
            plot_metrics = getattr(pl_module, f"{phase}_plot_metrics")
            current_epoch = trainer.current_epoch
            table = self.tables[phase]

            data = [current_epoch]
            for name, metric in plot_metrics.items():
                if "ConfusionMatrix" in name:
                    array = metric.compute().cpu().numpy()
                    index = list(range(array.shape[0]))
                    if self.num_to_labels is not None:
                        index = [self.num_to_labels[i] for i in index]

                    df_cm = pd.DataFrame(array, index=index, columns=index)
                    df_cm.index.name = "Actual"
                    df_cm.columns.name = "Predicted"
                    df_cm_normed = array / array.sum(axis=1, keepdims=True)

                    fig_, ax_ = plt.subplots(**self.fig_kws)
                    sns.heatmap(df_cm_normed, annot=df_cm, ax=ax_, **self.plot_kws)

                    # fig_, ax_ = pp_matrix(df_cm, **self.kwargs)
                else:
                    fig_, ax_ = metric.plot(**self.plot_kws)
                metric.reset()
                data.append(wandb.Image(fig_))
                plt.close(fig_)

            table.add_data(*data)
            key = f"{self.prefix}{phase}_plots"
            self.logger.log_table(key=key, columns=table.columns, data=table.data)

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.no_logger and (
            trainer.current_epoch % self.plot_every_n_epoch == 0 or trainer.current_epoch == trainer.max_epochs
        ):
            self.on_epoch_end_plotting(trainer, pl_module, phase="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.no_logger and (
            trainer.current_epoch % self.plot_every_n_epoch == 0 or trainer.current_epoch == trainer.max_epochs
        ):
            self.on_epoch_end_plotting(trainer, pl_module, phase="val")

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.no_logger:
            self.on_epoch_end_plotting(trainer, pl_module, phase="test")
