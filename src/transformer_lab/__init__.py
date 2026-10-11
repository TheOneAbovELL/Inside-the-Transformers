"""Inside the Transformer: a local, CPU-friendly mechanistic interpretability laboratory."""

__version__ = "0.1.0"

from .config import ExperimentConfig, ModelConfig, TaskConfig, TrainConfig  # noqa: E402
from .models import Transformer  # noqa: E402

__all__ = ["ExperimentConfig", "ModelConfig", "TaskConfig", "TrainConfig", "Transformer", "__version__"]
