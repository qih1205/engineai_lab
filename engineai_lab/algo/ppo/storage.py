from typing import Any


class RolloutStorage:
    """Lightweight container for storing rollout fragments."""

    def __init__(self) -> None:
        self.observations: list[Any] = []
        self.actions: list[Any] = []
        self.rewards: list[Any] = []
        self.dones: list[Any] = []

    def append(self, observation: Any, action: Any, reward: Any, done: Any) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.observations)
