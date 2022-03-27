import random
from collections import namedtuple
from typing import List, Optional, Dict

import torch
import gym
from gym import spaces
import numpy as np

import messenger
import messenger.envs.config as config
from messenger.envs.config import Entity

# Positions of the entities
Position = namedtuple('Position', ["x", "y"])


class MessengerEnv(gym.Env):
    '''
    Base Messenger class that defines the action and observation spaces.
    '''

    def __init__(self):
        super().__init__()
        # up, down, left, right, stay
        self.action_space = spaces.Discrete(len(config.ACTIONS))

        # observations, not including the text manual
        self.observation_space = spaces.Dict({
            "entities": spaces.Box(
                low=0,
                high=14,
                shape=(config.STATE_HEIGHT, config.STATE_WIDTH, 3),
                dtype=np.float32,
            ),
            "avatar": spaces.Box(
                low=15,
                high=16,
                shape=(config.STATE_HEIGHT, config.STATE_WIDTH, 1),
                dtype=np.float32,
            )
        })

        # FIXME: the following may affect globally
        np.set_printoptions(formatter={'int': self._numpy_formatter})

        # TODO: Move to the common operations about history from sub classes to this class, or implement as wrapper
        self._current_manual: Optional[List[str]] = None
        self._obs_history: List[Dict[str, torch.Tensor]] = []
        self._action_history: List[int] = []
        self._reward_history: List[str] = []

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)

    def render(self, mode='human'):
        str_repr = '\n'.join([
            self._get_terminal_clear_str(),
            self._get_instructions(),
            self._get_obs_str(self._obs_history),
            self._get_manual_str(self._current_manual),
            self._get_action_str(self._action_history),
            self._get_reward_str(self._reward_history),
        ])
        if mode == 'human':
            print(str_repr)
            return None
        elif mode == 'rgb_array':
            raise NotImplementedError()
        elif mode == 'ansi':
            return str_repr
        else:
            raise ValueError()

    def _numpy_formatter(self, i: int):
        ''' Format function passed to numpy print to make things pretty.
        '''
        id_map = {}
        for ent in messenger.envs.config.ALL_ENTITIES:
            id_map[ent.id] = ent.name[:2].upper()
        id_map[0] = '  '
        id_map[15] = 'A0'
        id_map[16] = 'AM'
        if i < 17:
            return id_map[i]
        else:
            return 'XX'

    def _get_instructions(self,) -> str:
        return '\n'.join([
            '\nMESSENGER\n',
            'Read the manual to get the message and bring it to the goal.',
            'A0 is you (agent) without the message, and AM is you with the message.',
            'The following is the symbol legend (symbol : entity)\n',
            '\n'.join([f'{ent.name[:2].upper()} : {ent.name}' for ent in messenger.envs.config.ALL_ENTITIES[:12]]),
            '\nNote when entities overlap the symbol might not make sense. Good luck!\n',
        ])

    def _get_obs_str(self, obs_history: List[Dict[str, torch.Tensor]]) -> str:
        if len(obs_history) == 0:
            return ''
        current_obs = obs_history[-1]
        grid = np.concatenate((current_obs['entities'], current_obs['avatar']), axis=-1)
        str_repr = str(np.sum(grid, axis=-1).astype('uint8'))
        return str_repr

    def _get_manual_str(self, manual: List[str]) -> str:
        man_str = f'Manual: {manual[0]}\n'
        for description in manual[1:]:
            man_str += f'        {description}\n'
        return man_str

    def _get_action_str(self, action_history: List[int]) -> str:
        action_map = {0: '^', 1: '_', 2: '<', 3: '>', 4: '*'}
        action_str_seq = ''.join([action_map[action] for action in action_history])
        return f'actions: {action_str_seq}'

    def _get_reward_str(self, reward_histroy: List[float]) -> str:
        if len(reward_histroy) == 0:
            return f'Reward:    current={"-":<4}    sum: {sum(reward_histroy):.2f}'
        else:
            return f'Reward:    current={reward_histroy[-1]:.2f}    sum: {sum(reward_histroy):.2f}'

    def _get_terminal_clear_str(self,) -> str:
        return '\033c\033[3J'


class Grid:
    '''
    Class which makes it easier to build a grid observation from the dict state
    return by VGDLEnv.
    '''

    def __init__(self, layers, shuffle=True):
        '''
        layers:
            Each add() operation will place a separate entity in a new layer.
            Thus, this is the upper-limit to the number of items to be added.
        shuffle:
            Place each items in a random order.
        '''
        self.grid = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, layers))
        self.order = list(range(layers))  # insertion order
        if shuffle:
            random.shuffle(self.order)
        self.layers = layers
        self.entity_count = 0

    def add(self, entity: Entity, position: Position):
        '''
        Add entity entity and position position.
        '''
        assert self.entity_count < self.layers, \
            f"Tried to add entity no. {self.entity_count} with {self.layers} layers."

        self.grid[position.y, position.x,
                  self.order[self.entity_count]] = entity.id
        self.entity_count += 1
