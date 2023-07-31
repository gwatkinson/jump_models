"""Defines callbacks that add logging capabilities to the training process
using the wandb logger."""

import logging

import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger

import wandb

logger = logging.getLogger(__name__)


class WandbTrainingCallback(Callback):
    def __init__(self, watch=True, watch_log="all", log_freq=100, log_graph=True):
        super().__init__()
        self.watch = watch
        self.log_freq = log_freq
        self.watch_log = watch_log
        self.log_graph = log_graph

        self.logger = None

    def setup(self, trainer, pl_module, stage=None):
        if self.logger is None:
            loggers = trainer.loggers

            for logger in loggers:
                if isinstance(logger, WandbLogger):
                    self.logger = logger
                    break

            if self.logger is None:
                logger.warning("No WandbLogger found. WandbCallback will not log anything.")
                self.watch = False

    def on_train_start(self, trainer, pl_module):
        if self.watch:
            self.logger.watch(pl_module, log=self.watch_log, log_freq=self.log_freq, log_graph=self.log_graph)

    def on_train_end(self, trainer, pl_module):
        if self.watch:
            self.logger.experiment.unwatch(pl_module)


class WandbOGBTrainingCallback(Callback):
    def __init__(self, watch=True, watch_log="all", log_freq=100, log_graph=True):
        super().__init__(watch=watch, watch_log=watch_log, log_freq=log_freq, log_graph=log_graph)
        self.tables = None
        self.num_figs = None

    def setup(self, trainer, pl_module, stage=None):
        super().setup(trainer, pl_module, stage=stage)

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

    def on_epoch_end_plotting(self, trainer, pl_module, phase="train"):
        if self.num_figs > 0:
            plot_metrics = getattr(pl_module, f"{phase}_plot_metrics")
            current_epoch = trainer.current_epoch
            table = self.tables[phase]

            data = [current_epoch]
            for _, metric in plot_metrics.items():
                fig_, ax_ = metric.plot()
                metric.reset()
                data.append(wandb.Image(fig_))
                plt.close(fig_)

            table.add_data(*data)
            self.logger.log_table(key=f"{phase}_plots", columns=table.columns, data=table.data, step=current_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end_plotting(trainer, pl_module, phase="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end_plotting(trainer, pl_module, phase="val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end_plotting(trainer, pl_module, phase="test")
