"""Small, explicit starting settings; these are not tuned performance claims."""

from dataclasses import asdict, dataclass

HEART_COMMIT = "ad122dba1e1d4c0aaee9bee8846ceba60ed680fe"
COMPONENTS = ("hearts", "queen", "self_moon", "opponent_moon")


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    heads: int = 2

    def __post_init__(self):
        if self.width < 4 or self.heads < 1 or self.width % self.heads:
            raise ValueError("width must be >= 4 and divisible by heads")


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    opponents: tuple[str, str, str] = ("easy", "medium", "hard")
    gamma: float = 1.0
    teacher_lambda: float = 0.75
    teacher_alpha: float = 0.01
    teacher_epsilon: float = 0.10
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ppo_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    batch_size: int = 64
    rollout_steps: int = 256
    ppo_epochs: int = 4

    def to_dict(self):
        return asdict(self)
