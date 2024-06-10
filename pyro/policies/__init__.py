"""Garage wrappers for gym environments."""

from pyro.policies.adaptive_argmax_policy import AdaptiveArgmaxPolicy
from pyro.policies.adaptive_gaussian_mlp_policy import AdaptiveGaussianMLPPolicy
<<<<<<< HEAD
from pyro.policies.adaptive_tanh_gaussian_policy import AdaptiveTanhGaussianPolicy
from pyro.policies.adaptive_gumbel_softmax_policy import AdaptiveGumbelSoftmaxPolicy
=======
from pyro.policies.adaptive_tanh_gaussian_policy import \
    AdaptiveTanhGaussianPolicy
>>>>>>> 86e044686651f01bd66c1063c70693c2645fd0b3
from pyro.policies.adaptive_toy_policy import AdaptiveToyPolicy
from pyro.policies.epsilon_greedy_policy import EpsilonGreedyPolicy
from pyro.policies.reproducing_policy import ReproducingPolicy

__all__ = [
    'AdaptiveArgmaxPolicy',
    'AdaptiveGaussianMLPPolicy',
    'AdaptiveTanhGaussianPolicy',
<<<<<<< HEAD
    'AdaptiveGumbelSoftmaxPolicy',
=======
>>>>>>> 86e044686651f01bd66c1063c70693c2645fd0b3
    'AdaptiveToyPolicy',
    'EpsilonGreedyPolicy',
    'ReproducingPolicy',
]
