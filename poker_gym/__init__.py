"""Poker environment for OpenAI Gym."""
from __future__ import annotations

from ray.tune.registry import register_env

from .envs import PokerEnv


def poker_env_creator(env_config):
    """Create a poker environment."""
    return PokerEnv(*env_config)


register_env(name="Poker-v0", env_creator=poker_env_creator)

__all__ = ["PokerEnv"]
