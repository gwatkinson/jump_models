"""Evaluator module that provides classes for evaluating models.

The main class is Evaluator, which is used to evaluate a model on a dataset. Usually with fine-tuning.
It requires a datamodule, a model with weights and a metric. The metric is used to evaluate the model.

Then, there is the EvaluatorList class, which is used to evaluate a list of models on a dataset.
This allows for easy comparison of models on a list of downstream tasks.
"""

from typing import List, Optional, Union

from lightning import LightningDataModule, LightningModule, Trainer

from src.utils.pylogger import get_pylogger

logger = get_pylogger(__name__)


class Evaluator:
    """Base class for an evaluator.

    This requires a pytorch lightning datamodule, a model with
    checkpoint weights, a metric and a Trainer.
    """

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
        """Finetune the model."""
        # logger.info(f"{self.prefix}Finetuning {self.model} on {self.datamodule}")
        self.trainer.fit(model=self.model, datamodule=self.datamodule)

    def evaluate(self):
        """Evaluate the model."""
        # logger.info(f"{self.prefix}Evaluating {self.model} on {self.datamodule}")
        self.trainer.test(model=self.model, datamodule=self.datamodule)
        return self.trainer.callback_metrics

    def visualize(self, outs, **kwargs):
        """Create visualizations."""
        # logger.info(f"{self.prefix}Visualizing {self.model} on {self.datamodule}")
        pass  # To implement in subclasses

    def run(self):
        """Run the evaluator."""
        self.finetune()
        outs = self.evaluate()
        self.visualize(outs, **self.visualize_kwargs)

    def __repr__(self):
        """Returns a string representation of the Evaluator object."""
        return f"{self.prefix}{self.__class__.__name__}"


class EvaluatorList:
    """Class containing a list of evaluators."""

    def __init__(self, evaluators: List[Evaluator], name: Optional[str] = None):
        self.evaluators = evaluators
        self.name = name
        self.prefix = f"({self.name}) " if self.name else ""

    def __getitem__(self, index: int) -> Evaluator:
        if len(self) == 0:
            raise IndexError("EvaluatorList is empty")
        return self.evaluators[index]

    def __len__(self) -> int:
        return len(self.evaluators)

    def __iter__(self):
        return iter(self.evaluators)

    def __add__(self, other: Union[Evaluator, "EvaluatorList", List[Evaluator]]):
        if isinstance(other, Evaluator):
            return EvaluatorList(self.evaluators + [other], name=self.name)
        elif isinstance(other, EvaluatorList):
            name = (self.name or "") + "+" + (other.name or "")
            return EvaluatorList(self.evaluators + other.evaluators, name=name)
        elif isinstance(other, list):
            return EvaluatorList(self.evaluators + other, name=self.name)
        else:
            raise TypeError(f"Cannot add {other} to {self}")

    def __radd__(self, other: Union[Evaluator, List[Evaluator]]):
        if isinstance(other, Evaluator):
            return EvaluatorList([other] + self.evaluators, name=self.name)
        elif isinstance(other, list):
            return EvaluatorList(other + self.evaluators, name=self.name)
        else:
            raise TypeError(f"Cannot add {other} to {self}")

    def append(self, other: Union[Evaluator, "EvaluatorList", List[Evaluator]]):
        return self.__add__(other)

    def __repr__(self):
        eval_str = "\n\t\t".join([evaluator.__repr__() for evaluator in self.evaluators])

        return f"""{self.prefix}{self.__class__.__name__}(
    n_evaluators={len(self.evaluators)},
    evaluators=
        {eval_str}
)"""

    def run(self):
        """Run the evaluators."""
        logger.info("Running EvaluationList")

        if len(self) == 0:
            logger.info("EvaluatorList is empty. Skipping.")
            return

        for evaluator in self.evaluators:
            try:
                evaluator.run()
            except Exception as e:
                logger.error(f"Error while running {evaluator}: {e}")
