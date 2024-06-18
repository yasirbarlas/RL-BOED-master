"""
Use Proximal Policy Optimisation to learn an agent that adaptively designs source location experiments
"""
import argparse

import joblib
import torch
import numpy as np

#from garage.experiment import deterministic
#from garage.torch import set_gpu_mode
from pyro import wrap_experiment, set_rng_seed
from pyro.algos import PPO
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import Trainer
from pyro.models.adaptive_experiment_model import CESModel
from pyro.policies import AdaptiveGaussianMLPPolicy
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
         src_filepath=None, discount=1., d=6, log_info=None,
         pi_lr=3e-4, vf_lr=3e-4, minibatch_size=4096, entropy_method="no_entropy",
         lr_clip_range=0.2, gae_lambda=0.97, policy_ent_coeff=0.01,
         center_adv=True, positive_adv=False, use_softplus_entropy=False,
         stop_entropy_gradient=False):
    if log_info is None:
        log_info = []

    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def ppo_source(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                   n_cont_samples=10, seed=0, src_filepath=None, discount=1.,
                   d=6, pi_lr=3e-4, vf_lr=3e-4, minibatch_size=4096, entropy_method="no_entropy",
                   lr_clip_range=0.2, gae_lambda=0.97, policy_ent_coeff=0.01,
                   center_adv=True, positive_adv=False, use_softplus_entropy=False,
                   stop_entropy_gradient=False):
        
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
            design_space = BatchBox(low=0.01, high=100, shape=(1, 1, 1, d))
            obs_space = BatchBox(low=torch.zeros((d+1,)),
                                 high=torch.as_tensor([100.] * d + [1.])
                                 )
            model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000,
                             n_elbo_samples=10)

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
                return AdaptiveGaussianMLPPolicy(
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
                      max_episode_length=budget,
                      policy_optimizer=policy_optimizer,
                      vf_optimizer=vf_optimizer,
                      num_train_per_epoch=1,
                      lr_clip_range=lr_clip_range,
                      discount=discount,
                      gae_lambda=gae_lambda,
                      center_adv=center_adv,
                      positive_adv=positive_adv,
                      policy_ent_coeff=policy_ent_coeff,
                      use_softplus_entropy=use_softplus_entropy,
                      stop_entropy_gradient=stop_entropy_gradient,
                      entropy_method=entropy_method)
                      
        trainer = Trainer(snapshot_config=ctxt)
        trainer.setup(algo=ppo, env=env)
        trainer.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    ppo_source(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
               n_cont_samples=n_cont_samples, seed=seed,
               src_filepath=src_filepath, discount=discount,
               d=d, pi_lr=pi_lr, vf_lr=vf_lr, minibatch_size=minibatch_size, entropy_method=entropy_method,
               lr_clip_range=lr_clip_range, gae_lambda=gae_lambda, policy_ent_coeff=policy_ent_coeff,
               center_adv=center_adv, positive_adv=positive_adv, use_softplus_entropy=use_softplus_entropy,
               stop_entropy_gradient=stop_entropy_gradient)

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
    parser.add_argument("--discount", default="0.99", type=float)
    parser.add_argument("--d", default="6", type=int)
    parser.add_argument("--pi-lr", default="3e-4", type=float)
    parser.add_argument("--vf-lr", default="3e-4", type=float)
    parser.add_argument("--minibatch-size", default="4096", type=int)
    parser.add_argument("--entropy-method", default="no_entropy", type=str.lower,
                        choices=["max", "regularized", "no_entropy"])
    parser.add_argument("--lr_clip_range", default="0.2", type=float)
    parser.add_argument("--gae_lambda", default="0.97", type=float)
    parser.add_argument("--policy_ent_coeff", default="0.00", type=float)
    parser.add_argument("--center_adv", default=False, type=str2bool)
    parser.add_argument("--positive_adv", default=False, type=str2bool)
    parser.add_argument("--use_softplus_entropy", default=False, type=str2bool)
    parser.add_argument("--stop_entropy_gradient", default=False, type=str2bool)

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
         d=args.d, log_info=log_info, pi_lr=args.pi_lr,
         vf_lr=args.vf_lr, minibatch_size=args.minibatch_size, entropy_method = args.entropy_method, 
         lr_clip_range=args.lr_clip_range, gae_lambda=args.gae_lambda, policy_ent_coeff=args.policy_ent_coeff,
         center_adv=args.center_adv, positive_adv=args.positive_adv, use_softplus_entropy=args.use_softplus_entropy,
         stop_entropy_gradient=args.stop_entropy_gradient)