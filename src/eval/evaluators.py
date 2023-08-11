"""Evaluator module that provides classes for evaluating models.

The main class is Evaluator, which is used to evaluate a model on a dataset. Usually with fine-tuning.
It requires a datamodule, a model with weights and a metric. The metric is used to evaluate the model.

Then, there is the EvaluatorList class, which is used to evaluate a list of models on a dataset.
This allows for easy comparison of models on a list of downstream tasks.
"""

import logging
from typing import List, Optional, Union

from lightning import LightningDataModule, LightningModule, Trainer

logger = logging.getLogger(__name__)


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
        finetune_kwargs: Optional[dict] = None,
        evaluate_kwargs: Optional[dict] = None,
        visualize_kwargs: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.name = name
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer

        self.finetune_kwargs = finetune_kwargs or {}
        self.evaluate_kwargs = evaluate_kwargs or {}
        self.visualize_kwargs = visualize_kwargs or {}

    def finetune(self, **kwargs):
        """Finetune the model."""
        logger.info(f"Finetuning {self.model} on {self.datamodule} with {self.trainer}")
        self.trainer.fit(model=self.model, datamodule=self.datamodule, **kwargs)

    def evaluate(self, **kwargs):
        """Evaluate the model."""
        logger.info(f"Evaluating {self.model} on {self.datamodule} with {self.trainer}")
        self.trainer.test(model=self.model, datamodule=self.datamodule, **kwargs)

    def visualize(self, **kwargs):
        """Create visualizations."""
        logger.info(f"Visualizing {self.model} on {self.datamodule} with {self.trainer}")
        pass  # To implement in subclasses

    def run(self):
        """Run the evaluator."""
        self.finetune(**self.finetune_kwargs)
        self.evaluate(**self.evaluate_kwargs)
        self.visualize(**self.visualize_kwargs)

    def __repr__(self):
        """Returns a string representation of the Evaluator object."""
        return f"""{"(" + str(self.name) + ") " if self.name else ""}{self.__class__.__name__}(
            datamodule={self.datamodule},
            model={self.model},
            trainer={self.trainer}
        )"""


class EvaluatorList:
    """Class containing a list of evaluators."""

    def __init__(self, evaluators: List[Evaluator], name: Optional[str] = None):
        self.evaluators = evaluators
        self.name = name

    def __getitem__(self, index: int) -> Evaluator:
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
        eval_str = "\n\t\t".join([str(evaluator) for evaluator in self.evaluators])

        return f"""({self.name}) {self.__class__.__name__}(
            n_evaluators={len(self.evaluators)},
            evaluators={eval_str}
        )"""

    def run(self):
        """Run the evaluators."""
        logger.info(f"Running {self}")
        for evaluator in self.evaluators:
            evaluator.run()
