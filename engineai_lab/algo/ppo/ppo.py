from typing import Any

from .storage import RolloutStorage


class PPOTrainer:
    """Simplified PPO trainer placeholder for future algorithm work."""

    def __init__(
        self,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

    def update(self, rollouts: RolloutStorage) -> dict[str, Any]:
        """Return summary statistics after a mock update."""
        metrics = {"samples": len(rollouts)}
        rollouts.clear()
        return metrics
