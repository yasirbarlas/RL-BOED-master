import argparse
import sys

import joblib
import pickle
import numpy as np
import torch

from time import time
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.util import set_seed

from pyro.models.adaptive_experiment_model import SourceModel, CESModel, DockingModel
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.spaces.batch_box import BatchBox

from pyro.algos import SUNRISE
from garage.torch.distributions import TanhNormal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_default_device(device)

def make_source_env(d, k, b, m, obs_sd, n_parallel, budget, n_cont_samples, bound_type, true_model=None):
    model = SourceModel(n_parallel=n_parallel, d=d, k=k, b=b, m=m, obs_sd=obs_sd)
    design_space = BatchBox(low=-4., high=4., shape=(1, 1, 1, d))
    obs_space = BatchBox(low=torch.as_tensor([-4.] * d + [-3.]), high=torch.as_tensor([4.] * d + [10.]))
    
    env = GymEnv(
            normalize(
                AdaptiveDesignEnv(
                    design_space, obs_space, model, budget,
                    n_cont_samples, true_model=true_model,
                    bound_type=bound_type),
                    normalize_obs=True))
    return env

def make_ces_env(d, obs_sd, n_parallel, budget, n_cont_samples, bound_type, true_model=None):
    model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000, n_elbo_samples=10, obs_sd=obs_sd)
    design_space = BatchBox(low=0.01, high=100, shape=(1, 1, 1, d))
    obs_space = BatchBox(low=torch.zeros((d+1,)), high=torch.as_tensor([100.] * d + [1.]))
    env = GymEnv(
            normalize(
                AdaptiveDesignEnv(
                    design_space, obs_space, model, budget,
                    n_cont_samples, true_model=true_model,
                    bound_type=bound_type),
                    normalize_obs=True))
    return env

def make_docking_env(d, n_parallel, budget, n_cont_samples, bound_type, true_model=None):
    model = DockingModel(n_parallel=n_parallel, d=d)
    design_space = BatchBox(low=-75., high=0., shape=(1, 1, 1, d))
    obs_space = BatchBox(low=torch.as_tensor([-75.] * d + [-70.]), high=torch.as_tensor([1.] * d + [2.]))
    env = GymEnv(
            normalize(
                AdaptiveDesignEnv(
                    design_space, obs_space, model, budget,
                    n_cont_samples, true_model=true_model,
                    bound_type=bound_type),
                    normalize_obs=True))
    return env

def main(src, results, dest, n_contrastive_samples, n_parallel,
         seq_length, edit_type, n_samples, seed, bound_type, env, source_d=2, source_k=2, 
         source_b=1e-1, source_m=1e-4, source_obs_sd=0.5, ces_d=6, ces_obs_sd=0.005, docking_d=1):
    set_seed(seed)
    if edit_type != 'a' and edit_type != 'w':
        sys.exit(f"inadmissible edit_type: {edit_type}")
    torch.set_printoptions(threshold=int(1e10))
    data = joblib.load(src)
    print(f"loaded data from {src}")
    if hasattr(data['algo'], '_sampler'):
        del data['algo']._sampler
    torch.cuda.empty_cache()
    algo = data['algo']

    if env.lower() == "source":
        env = make_source_env(source_d, source_k, source_b, source_m, source_obs_sd, n_parallel, seq_length, n_contrastive_samples, bound_type)
    elif env.lower() == "ces":
        env = make_ces_env(ces_d, ces_obs_sd, n_parallel, seq_length, n_contrastive_samples, bound_type)
    elif env.lower() == "docking":
        env = make_docking_env(docking_d, n_parallel, seq_length, n_contrastive_samples, bound_type)

    # If SUNRISE, we have (potentially) multiple policies
    if isinstance(algo, SUNRISE):
        pis = [pi.eval() for pi in algo._policies]
    else:
        pi = algo.policy
        pi.eval()

    # qf1, qf2 = algo._qf2, algo._qf2
    env.env.l = n_contrastive_samples
    env.env.n_parallel = n_parallel
    env.env.bound_type = bound_type
    rewards = []
    rep = n_samples // env.env.n_parallel
    print(f"{n_samples} / {env.env.n_parallel} = {rep} iterations to run")
    t0 = time()
    random = False
    if results is None:
        times = []
        for j in range(rep):
            print(f"iteration {j}")
            obs, _ = env.reset(n_parallel=n_parallel)
            # print("\n", env.env.theta0['theta'][0, 0], "\n")
            # print("\n", env.env.theta0['a'][0, 0], "\n")
            rewards.append([])
            for i in range(seq_length):
                mask = torch.ones_like(obs, dtype=torch.bool)[..., :1]
                ts = time()
                # Randomly choose a policy to use, and use the sampled action from that policy (generally best performance)
                #if isinstance(algo, SUNRISE):
                #    acts = []
                #    for i in range(len(pis)):
                #        actd, dist_info = pis[i].get_actions(obs, mask=mask)
                #        acts.append(actd)
                #    acti = np.random.choice(len(acts))
                #    act = acts[acti]
                # Combine policy distributions by taking the mean of the means, and the mean of the standard deviations
                # Combine as a TanhNormal distribution, and sample an action from that distribution (slightly better than below method)
                #if isinstance(algo, SUNRISE):
                #    means = []
                #    stds = []
                #    for i in range(len(pis)):
                #        _, dist_info = pis[i].get_actions(obs, mask=mask)
                #        means.append(dist_info["mean"])
                #        stds.append(dist_info["log_std"].exp())
                #    stacked_mean_tensors = torch.stack(means)
                #    stacked_std_tensors = torch.stack(stds)
                #    distrib = TanhNormal(torch.mean(stacked_mean_tensors, dim=0), torch.mean(stacked_std_tensors, dim=0))
                #    act = distrib.sample()
                # Calculate the mean of the policy distribution means, and use that as the action (as in paper, having worst performance)
                if isinstance(algo, SUNRISE):
                    means = []
                    for i in range(len(pis)):
                        _, dist_info = pis[i].get_actions(obs, mask=mask)
                        means.append(dist_info["mean"])
                    stacked_tensors = torch.stack(means)
                    act = torch.mean(stacked_tensors, dim=0)
                else:
                    act, dist_info = pi.get_actions(obs, mask=mask)
                te = time()
                times.append(te - ts)
                # exp_obs = lexpand(obs, n_parallel)
                # act, dist_info = pi.get_actions(exp_obs)
                # opt_index = torch.argmax(
                #     torch.min(qf1(exp_obs, act), qf2(exp_obs, act)),
                #     dim=0,
                #     keepdim=True)
                # opt_index = opt_index.expand((1,) + act.shape[1:])
                # act = torch.gather(act, 0, opt_index).squeeze()
                if random:
                    # act = env.action_space.sample((n_parallel,))/8.
                    low, high = env.action_space.bounds
                    act_dist = torch.distributions.Normal(
                        torch.zeros_like(low), torch.ones_like(high))
                    act = act_dist.sample((n_parallel,))/8.
                act = act.reshape(env.env.n_parallel, 1, 1, -1)
                # print(dist_info['logits'].topk(10))
                # print(f"act {act[0] * 4}")
                # print(f"mean {dist_info['mean'][0] * 4}")
                # print(f"std {dist_info['log_std'][0].exp() * 4}")
                es = env.step(act)
                obs, reward = es.observation, es.reward
                # obs_lb, obs_ub = env.observation_space.bounds
                # print(f"obs {obs[0][-1] * (obs_ub - obs_lb) + obs_lb}")
                # print(f"reward {reward[0]}")
                rewards[-1].append(reward)
            rewards[-1] = torch.stack(rewards[-1])
    else:
        with open(results, 'rb') as results_file:
            ys = []
            designs = []
            for i in range(seq_length):
                data = pickle.load(results_file)
                if i == 0:
                    if not hasattr(data['theta0'], "items"):
                        data['theta0'] = {'theta': data['theta0']}
                    theta0 = {k: v.cuda() for k, v in data['theta0'].items()}
                ys.append(data['y'].cuda())
                designs.append(data['d_star_design'].cuda())

            for j in range(rep):
                env.reset(n_parallel=n_parallel)
                for k, v in theta0.items():
                    env.env.thetas[k][0] = \
                        v[j * n_parallel:(j + 1) * n_parallel]
                rewards.append([])
                for i in range(seq_length):
                    y = ys[i]
                    design = designs[i]
                    reward = env.env.get_reward(
                        y[j*n_parallel:(j+1)*n_parallel],
                        design[j*n_parallel:(j+1)*n_parallel])
                    rewards[-1].append(reward)
                rewards[-1] = torch.stack(rewards[-1])
    times = np.array(times)
    print(f"mean time = {times.mean()}")
    print(f"se time = {times.std() / np.sqrt(len(times))}")
    rewards = torch.cat(rewards, dim=1)#.squeeze()
    print(rewards.shape)
    cumsum_rewards = torch.cumsum(rewards, dim=0)
    sum_rewards = torch.sum(rewards, dim=0)
    print(cumsum_rewards.shape)
    # print(cumsum_rewards.transpose(1, 0))
    t1 = time()
    print(f"compute time {t1-t0} seconds")
    print(f"saving results to {dest}")
    with open(dest, edit_type) as destfile:
        destfile.writelines("\n".join([
            src,
            str(sum_rewards.mean().item()),
            str(sum_rewards.std().item() / np.sqrt(sum_rewards.numel())),
            str(cumsum_rewards.transpose(1, 0))
        ]) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--results", default=None, type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--n_contrastive_samples", default=int(1e8), type=int)
    parser.add_argument("--n_parallel", default=1, type=int)
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--seq_length", default=20, type=int)
    parser.add_argument("--edit_type", default="w", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bound_type", default="lower", type=str.lower,
                        choices=["lower", "upper", "terminal"])
    parser.add_argument("--env", default="source", type=str)
    parser.add_argument("--source_d", default=2, type=int)
    parser.add_argument("--source_k", default=2, type=int)
    parser.add_argument("--source_b", default=1e-1, type=float)
    parser.add_argument("--source_m", default=1e-4, type=float)
    parser.add_argument("--source_obs_sd", default=0.5, type=float)
    parser.add_argument("--ces_d", default=6, type=int)
    parser.add_argument("--ces_obs_sd", default=0.005, type=float)
    parser.add_argument("--docking_d", default=1, type=int)
    args = parser.parse_args()
    bound_type = {
        "lower": LOWER, "upper": UPPER, "terminal": TERMINAL}[args.bound_type]
    main(args.src, args.results, args.dest, args.n_contrastive_samples,
         args.n_parallel, args.seq_length, args.edit_type, args.n_samples,
         args.seed, bound_type, env=args.env, source_d=args.source_d, source_k=args.source_k,
         source_b=args.source_b, source_m=args.source_m, source_obs_sd=args.source_obs_sd, ces_d=args.ces_d, 
         ces_obs_sd=args.ces_obs_sd, docking_d=args.docking_d)
