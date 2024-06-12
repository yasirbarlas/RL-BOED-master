"""RL algorithms."""
from pyro.algos.dqn import DQN
from pyro.algos.rem import REM
from pyro.algos.redq import REDQ
from pyro.algos.sbr import SBR
from pyro.algos.vpg import VPG
from pyro.algos.trpo import TRPO
from pyro.algos.ppo import PPO

__all__ = [
    'DQN',
    'REM',
    'REDQ',
    'SUNRISE',
    'SBR',
    'VPG',
    'TRPO',
    'PPO',
]
