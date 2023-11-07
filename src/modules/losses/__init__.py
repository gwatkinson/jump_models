# flake8: noqa: F403

from src.modules.losses.base_losses import CombinationLoss, LossWithTemperature, RegularizationLoss
from src.modules.losses.contrastive_losses import InfoNCE, NTXent, RegInfoNCE, RegNTXent
