"""The core of the environment."""
from __future__ import annotations

import gym
import numpy as np


class PokerAction:
    """The action space of the environment."""


class PokerObservation:
    """The observation space of the environment."""


class PokerInfo(dict):
    """The info object of the environment."""


class PokerEnv(gym.Env[PokerObservation, PokerAction]):
    """The poker environment."""

    metadata = {"render.modes": ["human"]}
    __slots__ = (
        "reward_range",
        "spec",
        "action_space",
        "observation_space",
        "_np_random",
        "__num_players",
        "__stack_size",
    )

    def __init__(self, num_players: int, stack_size: int):
        """Instantiate the environment."""
        self.__num_players = num_players
        self.__stack_size = stack_size

    def step(
        self, action: PokerAction
    ) -> tuple[PokerObservation, float, bool, PokerInfo]:
        """Step the environment's dynamics."""

    def reset(
        self,
        *,
        seed: int | None = None,
        return_info: bool = False,
        options: dict | None = None,
    ) -> PokerObservation | tuple[PokerObservation, PokerInfo]:
        """Reset the environment and return the initial observation."""

    def render(self, mode: str = "human") -> np.ndarray | str | None:
        """Render the environment."""

    @property
    def num_players(self) -> int:
        """Return the number of players."""
        return self.__num_players

    @property
    def stack_size(self) -> int:
        """Return the stack size."""
        return self.__stack_size
