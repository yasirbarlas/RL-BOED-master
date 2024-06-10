"""RL algorithms."""
from pyro.algos.dqn import DQN
from pyro.algos.rem import REM
<<<<<<< HEAD
from pyro.algos.redq import REDQ
from pyro.algos.sbr import SBR
from pyro.algos.vpg import VPG
from pyro.algos.trpo import TRPO
from pyro.algos.ppo import PPO
=======
from pyro.algos.sac import SAC
# from pyro.algos.vpg import VPG
# from pyro.algos.trpo import TRPO
>>>>>>> 86e044686651f01bd66c1063c70693c2645fd0b3

__all__ = [
    'DQN',
    'REM',
<<<<<<< HEAD
    'REDQ',
    'SUNRISE',
    'SBR',
    'VPG',
    'TRPO',
    'PPO'
]
=======
    'SAC',
    # 'VPG',
    # 'TRPO',
]

>>>>>>> 86e044686651f01bd66c1063c70693c2645fd0b3
