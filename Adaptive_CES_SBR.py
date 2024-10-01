"""
Use Scaled-By-Resetting to learn an agent that adaptively designs constant elasticity of
substitution experiments
"""

# Import libraries
import argparse

import joblib
import torch
import numpy as np

#from garage.experiment import deterministic
#from garage.torch import set_gpu_mode
from pyro import wrap_experiment, set_rng_seed
from pyro.algos import SBR
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import Trainer
from pyro.models.adaptive_experiment_model import CESModel
from pyro.policies import AdaptiveTanhGaussianPolicy
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction
from pyro.replay_buffer import PathBuffer
from pyro.sampler.local_sampler import LocalSampler
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from pyro.util import set_seed
from torch import nn
from dowel import logger

seeds = [373693, 943929, 675273, 79387, 508137, 557390, 756177, 155183, 262598,
         572185]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(n_parallel=1, budget=1, n_rl_itr=1, n_cont_samples=10, seed=0,
         log_dir=None, snapshot_mode="gap", snapshot_gap=500, bound_type=LOWER,
         src_filepath=None, discount=1., d=6, alpha=None, log_info=None,
         tau=5e-3, pi_lr=3e-4, qf_lr=3e-4, buffer_capacity=int(1e6), ens_size=2,
         M=2, minibatch_size=4096, reset_interval=256000, resets=True):
    if log_info is None:
        log_info = []

    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def sbr_ces(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                n_cont_samples=10, seed=0, src_filepath=None, discount=1.,
                d=6, alpha=None, tau=5e-3, pi_lr=3e-4, qf_lr=3e-4,
                buffer_capacity=int(1e6), ens_size=2, M=2,
                minibatch_size=4096, reset_interval=256000, resets=True):
        
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
            sbr = data["algo"]
            if not hasattr(sbr, "_sampler"):
                sbr._sampler = LocalSampler(agents=sbr.policy, envs=env,
                                            max_episode_length=budget,
                                            worker_class=VectorWorker)
            if not hasattr(sbr, "replay_buffer"):
                sbr.replay_buffer = PathBuffer(
                    capacity_in_transitions=buffer_capacity)
            if alpha is not None:
                sbr._use_automatic_entropy_tuning = False
                sbr._fixed_alpha = alpha
        else:
            logger.log("creating new policy")
            layer_size = 128
            design_space = BatchBox(low=0.01, high=100, shape=(1, 1, 1, d))
            obs_space = BatchBox(low=torch.zeros((d+1,)), high=torch.as_tensor([100.] * d + [1.]))
            model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000, n_elbo_samples=10, d=d)

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

            def make_q_func():
                return AdaptiveMLPQFunction(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=layer_size//2
                )

            env = make_env(design_space, obs_space, model, budget, n_cont_samples, bound_type)
            
            policy = make_policy()
            qfs = [make_q_func() for _ in range(ens_size)]
            replay_buffer = PathBuffer(capacity_in_transitions=buffer_capacity)
            sampler = LocalSampler(agents=policy, envs=env, max_episode_length=budget, worker_class=VectorWorker)

            sbr = SBR(env_spec=env.spec,
                      policy=policy,
                      qfs=qfs,
                      replay_buffer=replay_buffer,
                      sampler=sampler,
                      max_episode_length_eval=budget,
                      utd_ratio=64,
                      min_buffer_size=int(1e5),
                      target_update_tau=tau,
                      policy_lr=pi_lr,
                      qf_lr=qf_lr,
                      discount=discount,
                      discount_delta=0.,
                      fixed_alpha=alpha,
                      buffer_batch_size=minibatch_size,
                      reward_scale=1.,
                      M=M,
                      ent_anneal_rate=1/1.4e4,
                      reset_interval=reset_interval,
                      resets=resets)

        sbr.to()
        trainer = Trainer(snapshot_config=ctxt)
        trainer.setup(algo=sbr, env=env)
        trainer.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    sbr_ces(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
            n_cont_samples=n_cont_samples, seed=seed,
            src_filepath=src_filepath, discount=discount, d=d, alpha=alpha, tau=tau, pi_lr=pi_lr, qf_lr=qf_lr,
            buffer_capacity=buffer_capacity, ens_size=ens_size, M=M, 
            minibatch_size=minibatch_size, reset_interval=reset_interval, resets=resets)

    logger.dump_all()


if __name__ == "__main__":

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument("--alpha", default="-1", type=float)
    parser.add_argument("--d", default="6", type=int)
    parser.add_argument("--tau", default="5e-3", type=float)
    parser.add_argument("--pi-lr", default="3e-4", type=float)
    parser.add_argument("--qf-lr", default="3e-4", type=float)
    parser.add_argument("--buffer-capacity", default="1e6", type=float)
    parser.add_argument("--ens-size", default="2", type=int)
    parser.add_argument("--M", default="2", type=int)
    parser.add_argument("--minibatch-size", default="4096", type=int)
    parser.add_argument("--reset_interval", default="128", type=int)
    parser.add_argument("--resets", default=True, type=str2bool)
    args = parser.parse_args()
    bound_type_dict = {"lower": LOWER, "upper": UPPER, "terminal": TERMINAL}
    bound_type = bound_type_dict[args.bound_type]
    exp_id = args.id
    alpha = args.alpha if args.alpha >= 0 else None
    buff_cap = int(args.buffer_capacity)
    log_info = f"input params: {vars(args)}"
    main(n_parallel=args.n_parallel, budget=args.budget, n_rl_itr=args.n_rl_itr,
         n_cont_samples=args.n_contr_samples, seed=seeds[exp_id - 1],
         log_dir=args.log_dir, snapshot_mode=args.snapshot_mode,
         snapshot_gap=args.snapshot_gap, bound_type=bound_type,
         src_filepath=args.src_filepath, discount=args.discount, alpha=alpha,
         d=args.d, log_info=log_info, tau=args.tau, pi_lr=args.pi_lr,
         qf_lr=args.qf_lr, buffer_capacity=buff_cap, ens_size=args.ens_size,
         M=args.M, minibatch_size=args.minibatch_size, reset_interval=args.reset_interval, resets=args.resets)