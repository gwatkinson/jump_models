import my_code.data_modules  # noqa: F401
import my_code.models  # noqa: F401
import my_code.optimizers  # noqa: F401
from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI()


# Usage
# python main.py fit --model Model1 --data FakeDataset1 --optimizer LitAdam --lr_scheduler LitLRScheduler
# or python main.py fit --model my_code.models.Model1

# See help
# python main.py fit --model.help Model1
# python main.py fit --data.help FakeDataset2
# python main.py fit --optimizer.help Adagrad
# python main.py fit --lr_scheduler.help StepLR

if __name__ == "__main__":
    cli_main()
