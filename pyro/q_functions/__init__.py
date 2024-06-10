"""Adaptive versions of Garage q-functions"""

from pyro.q_functions.adaptive_discrete_q_function import \
    AdaptiveDiscreteQFunction
from pyro.q_functions.adaptive_dueling_q_function import \
    AdaptiveDuelingQFunction
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction
<<<<<<< HEAD
from pyro.q_functions.adaptive_lstm_q_function import AdaptiveLSTMQFunction
=======
>>>>>>> 86e044686651f01bd66c1063c70693c2645fd0b3

__all__ = [
    'AdaptiveDiscreteQFunction',
    'AdaptiveDuelingQFunction',
<<<<<<< HEAD
    'AdaptiveMLPQFunction',
    'AdaptiveLSTMQFunction'
=======
    'AdaptiveMLPQFunction'
>>>>>>> 86e044686651f01bd66c1063c70693c2645fd0b3
]
