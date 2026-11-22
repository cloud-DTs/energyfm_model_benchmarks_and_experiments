"""Define the local project paths and the scientific constants shared by E2/E3."""

from dataclasses import dataclass

CONTEXT = 512
HORIZON = 96
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [42, 43, 44]


@dataclass(frozen=True)
class TrainingSettings:
    """Record a training budget; inputs are explicit settings, output is immutable configuration."""

    epochs: int = 4
    batch_size: int = 16
    learning_rate: float = 1e-4
    scope: str = "last-block"
    dropout_mode: str = "adapted-only"
    gradient_clip: float | None = 1.0
    teacher_weight: float = 0.1
    weight_decay: float = 0.01
    evaluation_batch_size: int = 16
    independent_shuffle: bool = False


@dataclass(frozen=True)
class LocalUpdateSettings:
    """Store the settings of one short continual-learning update."""

    scope: str
    steps: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-5
    teacher_weight: float = 0.5
    weight_decay: float = 0.01


def comparison_settings(protocol, model_name, learning_rate=None):
    """Build the training configuration for one comparison candidate.

    Inputs: The protocol chooses matched or historical conditions. Model name selects
        historical settings, and the selected trial rate is used in fair mode.
    Outputs: A TrainingSettings record that can be planned, executed and saved with the
        candidate.
    Purpose: Keep the same full-model policy for fair runs and preserve the original
        schedules for historical reproduction.
    """
    if protocol == "fair":
        return TrainingSettings(
            scope="full",
            dropout_mode="disabled",
            learning_rate=learning_rate,
            independent_shuffle=True,
        )
    if model_name == "toto":
        return TrainingSettings(dropout_mode="all", gradient_clip=None)
    return TrainingSettings(epochs=3, learning_rate=1e-5)
