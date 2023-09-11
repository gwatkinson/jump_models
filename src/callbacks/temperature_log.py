from typing import List, Literal, Union

from lightning.pytorch.callbacks import Callback
from torch.nn import Parameter

from src.modules.layers.utils import _get_layer
from src.modules.losses import ClampedParameter
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


class TemperatureLoggingCallback(Callback):
    def __init__(
        self,
        attribute_name: Union[List[str], str] = ("criterion", "temperature"),
        key: str = "model/temperature",
        interval: Literal["epoch", "step"] = "step",
        frequency: int = 1,
    ):
        super().__init__()

        if isinstance(attribute_name, str):
            attribute_name = [attribute_name]

        self.attribute_name = attribute_name
        self.key = key
        self.interval = interval
        self.frequency = frequency

    def setup(self, trainer, pl_module, stage=None):
        try:
            self.get_temperature(pl_module)
            self.has_temperature = True
        except AttributeError:
            self.has_temperature = False
            log.info("No temperature found in model.")

    def get_temperature(self, pl_module):
        temperature = _get_layer(pl_module, self.attribute_name)
        if isinstance(temperature, ClampedParameter):
            temperature = temperature.value.item()
        elif isinstance(temperature, Parameter):
            temperature = temperature.item()
        return temperature

    def log_temperature(self, trainer, pl_module):
        if self.has_temperature:
            temperature = self.get_temperature(pl_module)
            for logger in trainer.loggers:
                logger.log_metrics({self.key: temperature})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.interval == "step" and (batch_idx % self.frequency == 0):
            self.log_temperature(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.interval == "epoch" and (trainer.current_epoch % self.frequency == 0):
            self.log_temperature(trainer, pl_module)
