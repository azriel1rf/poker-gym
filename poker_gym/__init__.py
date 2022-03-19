"""Poker environment for OpenAI Gym."""
from __future__ import annotations

from ray.rllib.env import PettingZooEnv
from ray.tune.registry import register_env

from .envs import poker_env


def poker_env_creator(env_config):
    """Create a poker environment."""
    environment = poker_env.env(**env_config)
    environment = PettingZooEnv(environment)
    return environment


register_env(name="Poker-v0", env_creator=poker_env_creator)

__all__ = ["poker_env"]
