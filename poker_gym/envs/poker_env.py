"""The core of the environment."""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Literal
from typing import Union

import numpy as np
from gym import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils import wrappers
from phevaluator import Card

NUM_CARDS = 52
BOARD_SIZE = 3

NUM_PLAYERS = 3
STARTING_CHIPS = 40

Mask = Literal[-1]
MASK: Mask = -1
CardWithMask = Union[Card, Mask]

CARD_SPACE = spaces.Discrete(NUM_CARDS + 1, start=-1)


class Action(Enum):
    """The action space of the environment."""

    CHECK_FOLD = 0
    CALL = 1
    BET_RAISE = 2


# class Info(dict):
#     """The info object of the environment."""


@dataclass
class Agent:
    """The agent of the environment."""

    player_id: int
    name: str = field(init=False)

    def __post_init__(self):
        """Initialize the agent."""
        self.name = f"player_{self.player_id}"

    def __hash__(self) -> int:
        """Return the hash of the agent."""
        return self.player_id


# @dataclass
# class StateByAgent:
#     """The state of the environment by agent."""

#     stack_size: int
#     is_active: bool
#     hole_card: tuple[CardWithMask, CardWithMask]

#     def retrieve_dict(self) -> dict:
#         """Retrieve the state by agent."""
#         result = {}
#         result["stack_size"] = self.stack_size
#         result["is_active"] = int(self.is_active)
#         result["hole_card"] = tuple(map(int, self.hole_card))
#         return result

#     def retrieve_space(self) -> spaces.Space:
#         """Retrieve the space of the state by agent."""
#         result = spaces.Dict()
#         result["stack_size"] = spaces.Discrete(STACK_SIZE * NUM_PLAYERS)
#         result["is_active"] = spaces.Discrete(2)
#         result["hole_card"] = spaces.Tuple([CARD_SPACE, CARD_SPACE])
#         return result


# @dataclass
# class State:
#     """The state of the environment."""

#     num_players: int
#     current_player: Agent
#     board_card: list[CardWithMask]
#     states_by_agent: dict[Agent, StateByAgent]

#     def retrieve_dict(self) -> dict:
#         """Retrieve the state."""
#         result: dict[str, Any] = {}
#         result["num_players"] = self.num_players
#         result["current_player"] = self.current_player.player_id

#         board_card = np.full(shape=3, fill_value=-1)
#         for i, card in enumerate(self.board_card):
#             board_card[i] = int(card)
#         result["board_card"] = board_card

#         result["states_by_agent"] = {
#             agent.name: state.retrieve_dict()
#             for agent, state in self.states_by_agent.items()
#         }
#         return result

#     def observe(self, agent: Agent) -> Observation:
#         """Observe the state."""
#         result = Observation(**deepcopy(self.__dict__))
#         for cur_agent, state in result.states_by_agent.items():
#             if cur_agent != agent:
#                 state.hole_card = (MASK, MASK)
#         return result


# @dataclass
# class Observation(State):
#     """The observation space of the environment."""

#     num_players: int
#     current_player: Agent
#     board_card: list[Card]
#     states_by_agent: dict[Agent, StateByAgent]

#     def retrieve_space(self) -> spaces.Space:
#         """Retrieve the space of the observation."""
#         result = spaces.Dict()
#         result["num_players"] = spaces.Discrete(NUM_PLAYERS)
#         result["current_player"] = spaces.Discrete(NUM_PLAYERS)
#         result["board_card"] = spaces.Tuple([CARD_SPACE] * BOARD_SIZE)
#         result["states_by_agent"] = spaces.Dict(
#             {
#                 agent.name: state.retrieve_space()
#                 for agent, state in self.states_by_agent.items()
#             }
#         )
#         return result


def env(num_players: int = NUM_PLAYERS, starting_chips: int = STARTING_CHIPS) -> AECEnv:
    """Create the environment."""
    environment = RawEnv(num_players=num_players, starting_chips=starting_chips)
    environment = wrappers.CaptureStdoutWrapper(environment)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class RawEnv(AECEnv):
    """The poker environment."""

    metadata = {"render.modes": ["human"]}

    possible_agents: list[Agent]
    agents: list[Agent]
    rewards: dict[Agent, float]
    dones: dict[Agent, bool]
    # infos: dict[Agent, dict]
    # observations: dict[Agent, dict]
    _cumulative_rewards: dict[Agent, float]
    _agent_selector: agent_selector
    agent_selection: Agent

    __num_players: int
    __starting_chips: int

    def __init__(self, num_players: int, starting_chips: int):
        """Instantiate the environment."""
        super().__init__()
        self.__num_players = num_players
        self.__starting_chips = starting_chips
        self.possible_agents = [Agent(i) for i in range(num_players)]
        self.action_spaces = {
            agent: spaces.Discrete(3) for agent in self.possible_agents
        }
        max_stack_size = starting_chips * num_players
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "stacking_chips": spaces.Box(
                        low=0, high=max_stack_size, shape=(num_players,), dtype=np.int8
                    ),
                    "is_active": spaces.MultiBinary(n=num_players),
                    "committed_chips": spaces.Box(
                        low=0, high=max_stack_size, shape=(num_players,), dtype=np.int8
                    ),
                    "button": spaces.Discrete(num_players),
                    "min_call_chips": spaces.Discrete(max_stack_size),
                    "min_bet_chips": spaces.Discrete(max_stack_size),
                }
            )
            for agent in self.possible_agents
        }
        self.reset()

    def step(self, action: Action) -> None:
        """Step the environment's dynamics."""
        agent = self.agent_selection
        if self.dones[agent]:
            self._was_done_step(action)
            return
        self._cumulative_rewards[agent] = 0

    def reset(self) -> None:
        """Reset the environment and return the initial observation."""
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_reward = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        # self.infos = {agent: Info() for agent in self.agents}
        # self.observations = {agent: Observation() for agent in self.agents}
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def render(self, mode: str = "human") -> np.ndarray | str | None:
        """Render the environment."""

    def observe(self, agent: Agent):
        """Return the current observation."""

    def state(self):
        """Return the current state."""

    @property
    def num_players(self) -> int:
        """Return the number of players."""
        return self.__num_players

    @property
    def starting_chips(self) -> int:
        """Return the starting chip stacks."""
        return self.__starting_chips
