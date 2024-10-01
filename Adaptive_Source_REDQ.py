"""
Use REDQ/DroQ to learn an agent that adaptively designs source location experiments
To use DroQ, enter the same number of M Q-functions into the algorithm (so M = 2 means 2 Q-functions),
enable dropout (at 0.01 or another probability), and lastly enable layer normalisation.
"""

# Import libraries
import argparse

import joblib
import torch
import numpy as np

#from garage.experiment import deterministic
#from garage.torch import set_gpu_mode
from pyro import wrap_experiment, set_rng_seed
from pyro.algos import REDQ
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import Trainer
from pyro.models.adaptive_experiment_model import SourceModel
from pyro.policies import AdaptiveTanhGaussianPolicy
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction
from pyro.q_functions.adaptive_lstm_q_function import AdaptiveLSTMQFunction
from pyro.replay_buffer import PathBuffer
from pyro.sampler.local_sampler import LocalSampler
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from pyro.util import set_seed
from torch import nn
from dowel import logger

# Seeds to train on (an agent can only be trained on one seed, this file allows you to select one of the seeds in the list)
seeds = [373693, 943929, 675273, 79387, 508137, 557390, 756177, 155183, 262598,
         572185]

# Get device for PyTorch (GPU or CPU for training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Main function for training based on a set of hyperparameters
def main(n_parallel=1, budget=1, n_rl_itr=1, n_cont_samples=10, seed=0,
         log_dir=None, snapshot_mode="gap", snapshot_gap=500, bound_type=LOWER,
         src_filepath=None, discount=1., alpha=None, k=2, d=2, log_info=None,
         tau=5e-3, pi_lr=3e-4, qf_lr=3e-4, buffer_capacity=int(1e6), ens_size=2,
         M=2, minibatch_size=4096, lstm_qfunction=False, dropout=0, 
         layer_normalization=False):
    if log_info is None:
        log_info = []
    # Function built underneath 'main()'
    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def redq_source(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                   n_cont_samples=10, seed=0, src_filepath=None, discount=1.,
                   alpha=None, k=2, d=2, tau=5e-3, pi_lr=3e-4, qf_lr=3e-4,
                   buffer_capacity=int(1e6), ens_size=2, M=2,
                   minibatch_size=4096, lstm_qfunction=False, dropout=0, 
                   layer_normalization=False):
        
        if log_info:
            logger.log(str(log_info))

        if torch.cuda.is_available():
            torch.set_default_device(device)
            logger.log("GPU available")
        else:
            logger.log("No GPU detected")

        set_seed(seed)
        set_rng_seed(seed)
        # If there is a saved agent to load
        if src_filepath:
            logger.log(f"loading data from {src_filepath}")
            data = joblib.load(src_filepath)
            env = data["env"]
            redq = data["algo"]
            if not hasattr(redq, "_sampler"):
                redq._sampler = LocalSampler(agents=redq.policy, envs=env,
                                            max_episode_length=budget,
                                            worker_class=VectorWorker)
            if not hasattr(redq, "replay_buffer"):
                redq.replay_buffer = PathBuffer(
                    capacity_in_transitions=buffer_capacity)
            if alpha is not None:
                redq._use_automatic_entropy_tuning = False
                redq._fixed_alpha = alpha
        
        # Train from scratch
        else:
            logger.log("creating new policy")
            # Choose number of nodes in the neural networks for each layer
            layer_size = 128
            # Construct the design/action and observation spaces
            design_space = BatchBox(low=-4., high=4., shape=(1, 1, 1, d))
            obs_space = BatchBox(low=torch.as_tensor([-4.] * d + [-3.]), high=torch.as_tensor([4.] * d + [10.]))
            # Choose the Location Finding statistical model (since this is for the Location Finding experimental design problem)
            model = SourceModel(n_parallel=n_parallel, d=d, k=k)

            # Function to make a Gymnasium environment of an experimental design problem
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

            # Function to make a policy
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

            # Function to make a Q-function
            def make_q_func():
                if lstm_qfunction == True:
                    return AdaptiveLSTMQFunction(
                        env_spec=env.spec,
                        encoder_sizes=[layer_size, layer_size],
                        encoder_output_nonlinearity=None,
                        emitter_sizes=[layer_size, layer_size],
                        emitter_output_nonlinearity=None,
                        encoding_dim=layer_size//2
                    )
                else:
                    return AdaptiveMLPQFunction(
                        env_spec=env.spec,
                        encoder_sizes=[layer_size, layer_size],
                        encoder_nonlinearity=nn.ReLU,
                        encoder_output_nonlinearity=None,
                        emitter_sizes=[layer_size, layer_size],
                        emitter_nonlinearity=nn.ReLU,
                        emitter_output_nonlinearity=None,
                        encoding_dim=layer_size//2,
                        dropout=dropout,
                        layer_normalization=layer_normalization
                    )

            # Make environment, policy, and Q-functions
            env = make_env(design_space, obs_space, model, budget, n_cont_samples, bound_type)
            
            policy = make_policy()
            qfs = [make_q_func() for _ in range(ens_size)]
            # Replay buffer for storing transitions
            replay_buffer = PathBuffer(capacity_in_transitions=buffer_capacity)
            # Sampler class for running workers (Garage-specific)
            sampler = LocalSampler(agents=policy, envs=env, max_episode_length=budget, worker_class=VectorWorker)

            # REDQ algorithm
            redq = REDQ(env_spec=env.spec,
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
                      ent_anneal_rate=1/1.4e4)

        redq.to()
        # Setup algorithm and environment (Garage-specific)
        trainer = Trainer(snapshot_config=ctxt)
        trainer.setup(algo=redq, env=env)
        # Start training using 'n_rl_itr' epochs
        trainer.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    # Run function for training
    redq_source(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
               n_cont_samples=n_cont_samples, seed=seed,
               src_filepath=src_filepath, discount=discount, alpha=alpha, k=k,
               d=d, tau=tau, pi_lr=pi_lr, qf_lr=qf_lr,
               buffer_capacity=buffer_capacity, ens_size=ens_size, M=M,
               minibatch_size=minibatch_size, lstm_qfunction=lstm_qfunction, 
               dropout=dropout, layer_normalization=layer_normalization)

    logger.dump_all()

# Parse arguments as required (hyperparameters, etc.)
if __name__ == "__main__":
    # https://stackoverflow.com/questions/715417/converting-from-a-string-to-boolean-in-python
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
    parser.add_argument("--d", default="2", type=int)
    parser.add_argument("--k", default="2", type=int)
    parser.add_argument("--tau", default="5e-3", type=float)
    parser.add_argument("--pi-lr", default="3e-4", type=float)
    parser.add_argument("--qf-lr", default="3e-4", type=float)
    parser.add_argument("--buffer-capacity", default="1e6", type=float)
    parser.add_argument("--ens-size", default="2", type=int)
    parser.add_argument("--M", default="2", type=int)
    parser.add_argument("--minibatch-size", default="4096", type=int)
    parser.add_argument("--lstm-q-function", default=False, type=str2bool)
    parser.add_argument("--layer-norm", default=False, type=str2bool)
    parser.add_argument("--dropout", default=0., type=float)
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
         k=args.k, d=args.d, log_info=log_info, tau=args.tau, pi_lr=args.pi_lr,
         qf_lr=args.qf_lr, buffer_capacity=buff_cap, ens_size=args.ens_size,
         M=args.M, minibatch_size=args.minibatch_size, lstm_qfunction=args.lstm_q_function,
         dropout=args.dropout, layer_normalization=args.layer_norm)