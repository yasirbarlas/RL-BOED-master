"""
Use ppo to learn an agent that adaptively designs source location experiments
"""
import argparse

import joblib
import torch
import numpy as np

#from garage.experiment import deterministic
#from garage.torch import set_gpu_mode
#from os import environ
from pyro import wrap_experiment, set_rng_seed
from pyro.algos import PPO
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import LocalRunner
from pyro.models.adaptive_experiment_model import SourceModel
from pyro.policies import AdaptiveTanhGaussianPolicy
from pyro.value_functions import AdaptiveMLPValueFunction
from pyro.sampler.local_sampler import LocalSampler
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from pyro.util import set_seed
from torch import nn
from dowel import logger

from garage.torch.optimizers import OptimizerWrapper

seeds = [373693, 943929, 675273, 79387, 508137, 557390, 756177, 155183, 262598,
         572185]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(n_parallel=1, budget=1, n_rl_itr=1, n_cont_samples=10, seed=0,
         log_dir=None, snapshot_mode="gap", snapshot_gap=500, bound_type=LOWER,
         src_filepath=None, discount=1., k=2, d=2, log_info=None,
         pi_lr=3e-4, vf_lr=3e-4, minibatch_size=4096):
    if log_info is None:
        log_info = []

    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def ppo_source(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                   n_cont_samples=10, seed=0, src_filepath=None, discount=1.,
                   k=2, d=2, pi_lr=3e-4, vf_lr=3e-4, minibatch_size=4096):
        
        if log_info:
            logger.log(str(log_info))

        if torch.cuda.is_available():
            torch.set_default_device(device)
            logger.log("GPU available")
        else:
            logger.log("No GPU detected")

        set_seed(seed)
        set_rng_seed(seed)
        # if there is a saved agent to load
        if src_filepath:
            logger.log(f"loading data from {src_filepath}")
            data = joblib.load(src_filepath)
            env = data["env"]
            ppo = data["algo"]
        else:
            logger.log("creating new policy")
            layer_size = 128
            design_space = BatchBox(low=-4., high=4., shape=(1, 1, 1, d))
            obs_space = BatchBox(low=torch.as_tensor([-4.] * d + [-3.]),
                                 high=torch.as_tensor([4.] * d + [10.])
                                 )
            model = SourceModel(n_parallel=n_parallel, d=d, k=k)

            def make_env(design_space, obs_space, model, budget, n_cont_samples,
                         bound_type, true_model=None):
                env = GymEnv(
                    normalize(
                        AdaptiveDesignEnv(
                            design_space, obs_space, model, budget,
                            n_cont_samples, true_model=true_model,
                            bound_type=bound_type),
                        normalize_obs=True
                    )
                )
                return env

            def make_policy():
                return AdaptiveTanhGaussianPolicy(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=layer_size//2,
                    init_std=np.sqrt(1 / 3),
                    min_std=np.exp(-20.),
                    max_std=np.exp(0.),
                )

            def make_v_func():
                return AdaptiveMLPValueFunction(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.Tanh,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.Tanh,
                    emitter_output_nonlinearity=None,
                    encoding_dim=16
                )

            env = make_env(design_space, obs_space, model, budget,
                           n_cont_samples, bound_type)

            policy = make_policy()
            value_function = make_v_func()
            
            policy_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=pi_lr)),
                policy,
                max_optimization_epochs=10,
                minibatch_size=minibatch_size)
            
            vf_optimizer = OptimizerWrapper(
                (torch.optim.Adam, dict(lr=vf_lr)),
                value_function,
                max_optimization_epochs=10,
                minibatch_size=minibatch_size)

            sampler = LocalSampler(agents=policy, envs=env,
                                   max_episode_length=budget,
                                   worker_class=VectorWorker)

            ppo = PPO(env_spec=env.spec,
                      policy=policy,
                      value_function=value_function,
                      sampler=sampler,
                      policy_optimizer=policy_optimizer,
                      vf_optimizer=vf_optimizer,
                      lr_clip_range=2e-1,
                      discount=discount,
                      gae_lambda=0.97,
                      center_adv=True,
                      positive_adv=False,
                      policy_ent_coeff = 0.0,
                      use_softplus_entropy = False,
                      stop_entropy_gradient = False,
                      entropy_method = "no_entropy")
                      

        runner = LocalRunner(snapshot_config=ctxt)
        runner.setup(algo=ppo, env=env, sampler_cls=LocalSampler,
                     worker_class=VectorWorker)
        runner.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    ppo_source(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
               n_cont_samples=n_cont_samples, seed=seed,
               src_filepath=src_filepath, discount=discount, k=k,
               d=d, pi_lr=pi_lr, vf_lr=vf_lr, minibatch_size=minibatch_size)

    logger.dump_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1", type=int)
    parser.add_argument("--n-parallel", default="100", type=int)
    parser.add_argument("--budget", default="30", type=int)
    parser.add_argument("--n-rl-itr", default="50", type=int)
    parser.add_argument("--n-contr-samples", default="10", type=int)
    parser.add_argument("--log-dir", default=None, type=str)
    parser.add_argument("--src-filepath", default=None, type=str)
    parser.add_argument("--snapshot-mode", default="gap", type=str)
    parser.add_argument("--snapshot-gap", default=500, type=int)
    parser.add_argument("--bound-type", default="terminal", type=str.lower,
                        choices=["lower", "upper", "terminal"])
    parser.add_argument("--discount", default="1", type=float)
    parser.add_argument("--d", default="2", type=int)
    parser.add_argument("--k", default="2", type=int)
    parser.add_argument("--pi-lr", default="3e-4", type=float)
    parser.add_argument("--qf-lr", default="3e-4", type=float)
    parser.add_argument("--minibatch-size", default="4096", type=int)
    args = parser.parse_args()
    bound_type_dict = {"lower": LOWER, "upper": UPPER, "terminal": TERMINAL}
    bound_type = bound_type_dict[args.bound_type]
    exp_id = args.id
    log_info = f"input params: {vars(args)}"
    main(n_parallel=args.n_parallel, budget=args.budget, n_rl_itr=args.n_rl_itr,
         n_cont_samples=args.n_contr_samples, seed=seeds[exp_id - 1],
         log_dir=args.log_dir, snapshot_mode=args.snapshot_mode,
         snapshot_gap=args.snapshot_gap, bound_type=bound_type,
         src_filepath=args.src_filepath, discount=args.discount,
         k=args.k, d=args.d, log_info=log_info, pi_lr=args.pi_lr,
         vf_lr=args.vf_lr, minibatch_size=args.minibatch_size)