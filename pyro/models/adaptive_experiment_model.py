from abc import ABC

from contextlib import ExitStack
from functools import partial
from pyro import poutine
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv
from pyro.util import is_bad
from torchdiffeq import odeint

import pyro
import pyro.distributions as dist
import torch
import math

EPS = 2**-22


class ExperimentModel(ABC):
    """
    Basic interface for probabilistic models
    """

    def __init__(self):
        self.epsilon = torch.tensor(EPS)

    def sanity_check(self):
        assert self.var_dim > 0
        assert len(self.var_names) > 0

    def make_model(self):
        raise NotImplementedError

    def reset(self, n_parallel):
        raise NotImplementedError

    def run_experiment(self, design, theta):
        """
        Execute an experiment with given design.
        """
        # create model from sampled params
        cond_model = pyro.condition(self.make_model(), data=theta)

        # infer experimental outcome given design and model
        y = cond_model(design)
        y = y.detach().clone()
        return y

    def get_likelihoods(self, y, design, thetas):
        size = thetas[self.var_names[0]].shape[0]
        cond_dict = dict(thetas)
        cond_dict.update({self.obs_label: lexpand(y, size)})
        cond_model = pyro.condition(self.make_model(), data=cond_dict)
        trace = poutine.trace(cond_model).get_trace(lexpand(design, size))
        trace.compute_log_prob()
        likelihoods = trace.nodes[self.obs_label]["log_prob"]
        return likelihoods

    def sample_theta(self, num_theta):
        dummy_design = torch.zeros(
            (num_theta, self.n_parallel, 1, 1, self.var_dim))
        cur_model = self.make_model()
        trace = poutine.trace(cur_model).get_trace(dummy_design)
        thetas = dict([(l, trace.nodes[l]["value"]) for l in self.var_names])
        return thetas


class CESModel(ExperimentModel):
    def __init__(self, init_rho_model=None, init_alpha_model=None,
                 init_mu_model=None, init_sig_model=None, n_parallel=1,
                 obs_sd=0.005, obs_label="y", n_elbo_samples=100,
                 n_elbo_steps=100, elbo_lr=0.04, d=6):
        super().__init__()
        self.init_rho_model = init_rho_model if init_rho_model is not None \
            else torch.ones(n_parallel, 1, 2)
        self.init_alpha_model = init_alpha_model \
            if init_alpha_model is not None else torch.ones(n_parallel, 1, 3)
        self.init_mu_model = init_mu_model if init_mu_model is not None \
            else torch.ones(n_parallel, 1)
        self.init_sig_model = init_sig_model if init_sig_model is not None \
            else 3. * torch.ones(n_parallel, 1)
        self.rho_con_model = self.init_rho_model.detach().clone()
        self.alpha_con_model = self.init_alpha_model.detach().clone()
        self.u_mu_model = self.init_mu_model.detach().clone()
        self.u_sig_model = self.init_sig_model.detach().clone()
        self.n_parallel, self.elbo_lr = n_parallel, elbo_lr
        self.n_elbo_samples, self.n_elbo_steps = n_elbo_samples, n_elbo_steps
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.param_names = [
            "rho_con",
            "alpha_con",
            "u_mu",
            "u_sig",
        ]
        self.var_names = ["rho", "alpha", "u"]
        self.var_dim = d
        self.sanity_check()

    def reset(self, init_rho_model=None, init_alpha_model=None,
              init_mu_model=None, init_sig_model=None, n_parallel=None):
        if n_parallel is not None:
            self.n_parallel = n_parallel
            self.init_rho_model = init_rho_model if init_rho_model \
                else torch.ones(self.n_parallel, 1, 2)
            self.init_alpha_model = init_alpha_model if init_alpha_model \
                else torch.ones(self.n_parallel, 1, 3)
            self.init_mu_model = init_mu_model if init_mu_model \
                else torch.ones(self.n_parallel, 1)
            self.init_sig_model = init_sig_model if init_sig_model \
                else 3. * torch.ones(self.n_parallel, 1)
            self.rho_con_model = self.init_rho_model.detach().clone()
            self.alpha_con_model = self.init_alpha_model.detach().clone()
            self.u_mu_model = self.init_mu_model.detach().clone()
            self.u_sig_model = self.init_sig_model.detach().clone()

    def make_model(self):
        def model(design):
            # pyro.set_rng_seed(10)
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            #print("design", design.shape)
            #print("batchshape", batch_shape)
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                rho_shape = batch_shape + (self.rho_con_model.shape[-1],)
                #print("rhoshape", rho_shape)
                rho = 0.01 + 0.99 * pyro.sample(
                    "rho",
                    dist.Dirichlet(self.rho_con_model.expand(rho_shape))
                ).select(-1, 0)
                #print("rhomodelshape", self.rho_con_model.shape)
                #print("rho on its own shape", rho.shape)
                alpha_shape = batch_shape + (self.alpha_con_model.shape[-1],)
                alpha = pyro.sample(
                    "alpha",
                    dist.Dirichlet(self.alpha_con_model.expand(alpha_shape))
                )
                u = pyro.sample(
                    "u",
                    dist.LogNormal(
                        self.u_mu_model.expand(batch_shape),
                        self.u_sig_model.expand(batch_shape)
                    )
                )
                #print("u", u.shape)
                rho = rexpand(rho, design.shape[-2])
                #print("rhoexpand", rho.shape)
                u = rexpand(u, design.shape[-2])
                #print("uexpand.shape", u.shape)
                #print("rho.unsqueeze(-1)", rho.unsqueeze(-1).shape)
                d1, d2 = design[..., 0:math.floor(self.var_dim/2)], design[..., math.floor(self.var_dim/2):self.var_dim]
                #print(d1.shape, d2.shape)
                u1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
                u2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
                mean = u * (u1rho - u2rho)
                sd = u * self.obs_sd * (
                        1 + torch.norm(d1 - d2, dim=-1, p=2))
                #print("m,sd", mean.shape, sd.shape)
                emission_dist = dist.CensoredSigmoidNormal(
                    mean, sd, 1 - self.epsilon, self.epsilon
                ).to_event(1)
                #print("emission_dist", emission_dist.shape)
                y = pyro.sample(self.obs_label, emission_dist)
                #print("y", y.shape)
                return y

        return model

    def get_params(self):
        return torch.cat(
            [
                self.rho_con_model.reshape(self.n_parallel, -1),
                self.alpha_con_model.reshape(self.n_parallel, -1),
                self.u_mu_model.reshape(self.n_parallel, -1),
                self.u_sig_model.reshape(self.n_parallel, -1),
            ],
            dim=-1
        )


def holling2(a, th, t, n):
    an = a * n
    return -an / (1 + an * th)


def holling3(a, th, t, n):
    an2 = a * n * n
    return -an2 / (1 + an2 * th)


class PreyModel(ExperimentModel):
    def __init__(self, a_mu=None, a_sig=None, th_mu=None, th_sig=None, tau=24.,
                 n_parallel=1, obs_sd=0.005, obs_label="y"):
        super().__init__()
        self.a_mu = a_mu if a_mu is not None \
            else torch.ones(n_parallel, 1, 1) * -1.4
        self.a_sig = a_sig if a_sig is not None \
            else torch.ones(n_parallel, 1, 1) * 1.35
        self.th_mu = th_mu if th_mu is not None \
            else torch.ones(n_parallel, 1, 1) * -1.4
        self.th_sig = th_sig if th_sig is not None \
            else torch.ones(n_parallel, 1, 1) * 1.35
        self.tau = tau
        self.n_parallel = n_parallel
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.var_names = ["a", "th"]
        self.var_dim = 1
        self.sanity_check()

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            design = design.float()
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                a_shape = batch_shape + self.a_mu.shape[-1:]
                a = pyro.sample(
                    "a",
                    dist.LogNormal(
                        self.a_mu.expand(a_shape),
                        self.a_sig.expand(a_shape)
                    ).to_event(1)
                )
                a = a.expand(a.shape[:-1] + design.shape[-2:-1])
                th_shape = batch_shape + self.th_mu.shape[-1:]
                th = pyro.sample(
                    "th",
                    dist.LogNormal(
                        self.th_mu.expand(th_shape),
                        self.th_sig.expand(th_shape)
                    ).to_event(1)
                )
                th = th.expand(th.shape[:-1] + design.shape[-2:-1])
                diff_func = partial(
                    holling3,
                    a.flatten(),
                    th.flatten())
                int_sol = odeint(
                    diff_func,
                    design.flatten(),
                    torch.tensor([0., self.tau]),
                    method="rk4",
                    options={'step_size': 1.})
                n_t = int_sol[-1].reshape(design.shape)
                p_t = (design - n_t) / design
                emission_dist = dist.Binomial(design.reshape(a.shape),
                                              p_t.reshape(a.shape), validate_args=False).to_event(1)
                #print("design.shape", design.shape, design)
                #print("design.reshape(a.shape)", design.reshape(a.shape).shape, design.reshape(a.shape))
                #print("a.shape", a.shape)
                #print("p_t.shape", p_t.shape, p_t)
                #print("design.reshape(a.shape)", p_t.reshape(a.shape).shape, p_t.reshape(a.shape))
                #print("emission_dist.shape", emission_dist.shape)
                n = pyro.sample(
                    self.obs_label, emission_dist
                )
                return n

        return model

    def reset(self, n_parallel):
        self.n_parallel = n_parallel
        self.a_mu = torch.ones(n_parallel, 1, 1) * -1.4
        self.a_sig = torch.ones(n_parallel, 1, 1) * 1.35
        self.th_mu = torch.ones(n_parallel, 1, 1) * -1.4
        self.th_sig = torch.ones(n_parallel, 1, 1) * 1.35


class SourceModel(ExperimentModel):
    def __init__(self, d=2, k=2, theta_mu=None, theta_sig=None, alpha=None,
                 b=1e-1, m=1e-4, n_parallel=1, obs_sd=0.5, obs_label="y"):
        super().__init__()
        self.theta_mu = theta_mu if theta_mu is not None \
            else torch.zeros(n_parallel, 1, k, d)
        self.theta_sig = theta_sig if theta_sig is not None \
            else torch.ones(n_parallel, 1, k, d)
        self.alpha = alpha if alpha is not None \
            else torch.ones(n_parallel, 1, k)
        self.d, self.k, self.b, self.m = d, k, b, m
        self.obs_sd, self.obs_label = obs_sd, obs_label
        self.n_parallel = n_parallel
        self.var_names = ["theta"]
        self.var_dim = d
        self.sanity_check()

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                theta_shape = batch_shape + self.theta_mu.shape[-2:]
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.theta_mu.expand(theta_shape),
                        self.theta_sig.expand(theta_shape)
                    ).to_event(2)
                )
                #print("theta", theta.shape)
                distance = torch.square(theta - design).sum(dim=-1)
                ratio = self.alpha / (self.m + distance)
                mu = self.b + ratio.sum(dim=-1, keepdims=True)
                emission_dist = dist.Normal(
                    torch.log(mu), self.obs_sd
                ).to_event(1)
                #print("emission_dist", emission_dist.shape)
                y = pyro.sample(self.obs_label, emission_dist)
                #print("y", y.shape)
                return y

        return model

    def reset(self, n_parallel):
        self.n_parallel = n_parallel
        self.theta_mu = torch.zeros(n_parallel, 1, self.k, self.d)
        self.theta_sig = torch.ones(n_parallel, 1, self.k, self.d)
        self.alpha = torch.ones(n_parallel, 1, self.k)

###################

def sigmoid(x, top, bottom, ee50, slope):
    return (top - bottom) * torch.sigmoid((x - ee50) * slope) + bottom

class DockingModel(ExperimentModel):
    def __init__(self, n_parallel, d=1, init_top_prior_con=None, init_bottom_prior_con=None, init_ee50_prior_mu=None, init_ee50_prior_sd=None, init_slope_prior_mu=None,
            init_slope_prior_sd=None, obs_label="y"):
        super().__init__()
        self.init_top_prior_con = init_top_prior_con if init_top_prior_con is not None \
                    else torch.tensor([25., 75.])
        self.init_bottom_prior_con = init_bottom_prior_con if init_bottom_prior_con is not None \
                    else torch.tensor([4., 96.])
                
        self.init_ee50_prior_mu = init_ee50_prior_mu if init_ee50_prior_mu is not None \
                    else torch.tensor(-50.)
        self.init_ee50_prior_sd = init_ee50_prior_sd if init_ee50_prior_sd is not None \
                    else torch.tensor(15.)
                
        self.init_slope_prior_mu = init_slope_prior_mu if init_slope_prior_mu is not None \
                    else torch.tensor(-0.15)
        self.init_slope_prior_sd = init_slope_prior_sd if init_slope_prior_sd is not None \
                    else torch.tensor(0.1)
                
        self.top_prior_con = self.init_top_prior_con.detach().clone()
        self.bottom_prior_con = self.init_bottom_prior_con.detach().clone()
        self.ee50_prior_mu = self.init_ee50_prior_mu.detach().clone()
        self.ee50_prior_sd = self.init_ee50_prior_sd.detach().clone()
        self.slope_prior_mu = self.init_slope_prior_mu.detach().clone()
        self.slope_prior_sd = self.init_slope_prior_sd.detach().clone()

        #top_prior_con_1 = 25. * torch.ones(n_parallel, 1)
        #top_prior_con_2 = 75. * torch.ones(n_parallel, 1)
        #bottom_prior_con_1 = 4. * torch.ones(n_parallel, 1)
        #bottom_prior_con_2 = 96. * torch.ones(n_parallel, 1)

        #top_prior_con = torch.ones(n_parallel, 1, 2)
        #top_prior_con[..., 0] = 25
        #top_prior_con[..., 1] = 75

        #bottom_prior_con = torch.ones(n_parallel, 1, 2)
        #bottom_prior_con[..., 0] = 25
        #bottom_prior_con[..., 1] = 75

        #ee50_prior_mu, ee50_prior_sd = -50. * torch.ones(n_parallel, 1), 15. * torch.ones(n_parallel, 1)
        #slope_prior_mu, slope_prior_sd = -0.15 * torch.ones(n_parallel, 1), 0.1 * torch.ones(n_parallel, 1)

        #self.top_prior_con = top_prior_con
        #self.bottom_prior_con = bottom_prior_con
        #self.ee50_prior_mu = ee50_prior_mu
        #self.ee50_prior_sd = ee50_prior_sd
        #self.slope_prior_mu = slope_prior_mu
        #self.slope_prior_sd = slope_prior_sd
        
        self.obs_label = obs_label
        self.n_parallel = n_parallel
        self.var_names = ["top", "bottom", "ee50", "slope"]
        self.var_dim = d
        self.sanity_check()

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            #print("batchshape", batch_shape)
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                top_shape = batch_shape + (self.top_prior_con.shape[-1],)
                top = pyro.sample("top", dist.Dirichlet(self.top_prior_con.expand(top_shape))).select(-1, 0)
                #print("self.top_prior_con", self.top_prior_con.shape)
                bottom_shape = batch_shape + (self.bottom_prior_con.shape[-1],)
                bottom = pyro.sample("bottom", dist.Dirichlet(self.bottom_prior_con.expand(bottom_shape))).select(-1, 0)
                #print("self.bottom_prior_con", self.bottom_prior_con.shape)
                ee50 = pyro.sample("ee50", dist.Normal(self.ee50_prior_mu.expand(batch_shape), self.ee50_prior_sd.expand(batch_shape)))
                #print("self.ee50_prior_mu", self.ee50_prior_mu.shape)
                slope = pyro.sample("slope", dist.Normal(self.slope_prior_mu.expand(batch_shape), self.slope_prior_sd.expand(batch_shape)))
                #print("self.slope_prior_mu", self.slope_prior_mu.shape)
                #print("topshape", top_shape, "bottomshape", bottom_shape)
                #print("des", design.shape, "top", top.shape, "bottom", bottom.shape, "ee50", ee50.shape, "slope", slope.shape)
                top = rexpand(top, design.shape[-2])
                bottom = rexpand(bottom, design.shape[-2])
                ee50 = rexpand(ee50, design.shape[-2])
                slope = rexpand(slope, design.shape[-2])
                a = design.squeeze(-2).shape
                #print("a", a)
                #print("topexpand", top.shape)
                #print("bottomexpand", bottom.shape)
                #print("ee50expand", ee50.shape)
                #print("slopeexpand", slope.shape)
                top = top.unsqueeze(-1)
                bottom = bottom.unsqueeze(-1)
                ee50 = ee50.unsqueeze(-1)
                slope = slope.unsqueeze(-1)
                hit_rate = sigmoid(design, top, bottom, ee50, slope)
                #print(hit_rate, hit_rate.shape)
                emission_dist = dist.Bernoulli(hit_rate.reshape(a)).to_event(1)
                #print("emission_dist", emission_dist.shape)
                #print(emission_dist.sample())
                y = pyro.sample(self.obs_label, emission_dist)
                #print("design", design, design.shape)
                #print("y", y, y.shape)
                return y

        return model
    
    def reset(self, n_parallel, init_top_prior_con=None, init_bottom_prior_con=None, init_ee50_prior_mu=None, init_ee50_prior_sd=None, init_slope_prior_mu=None,
            init_slope_prior_sd=None):
            if n_parallel is not None:
                self.init_top_prior_con = init_top_prior_con if init_top_prior_con is not None \
                    else torch.tensor([25., 75.])
                self.init_bottom_prior_con = init_bottom_prior_con if init_bottom_prior_con is not None \
                    else torch.tensor([4., 96.])
                
                self.init_ee50_prior_mu = init_ee50_prior_mu if init_ee50_prior_mu is not None \
                    else torch.tensor(-50.)
                self.init_ee50_prior_sd = init_ee50_prior_sd if init_ee50_prior_sd is not None \
                    else torch.tensor(15.)
                
                self.init_slope_prior_mu = init_slope_prior_mu if init_slope_prior_mu is not None \
                    else torch.tensor(-0.15)
                self.init_slope_prior_sd = init_slope_prior_sd if init_slope_prior_sd is not None \
                    else torch.tensor(0.1)
                
                self.top_prior_con = self.init_top_prior_con.detach().clone()
                self.bottom_prior_con = self.init_bottom_prior_con.detach().clone()
                self.ee50_prior_mu = self.init_ee50_prior_mu.detach().clone()
                self.ee50_prior_sd = self.init_ee50_prior_sd.detach().clone()
                self.slope_prior_mu = self.init_slope_prior_mu.detach().clone()
                self.slope_prior_sd = self.init_slope_prior_sd.detach().clone()

                #top_prior_con = torch.ones(n_parallel, 1, 2)
                #top_prior_con[..., 0] = 25
                #top_prior_con[..., 1] = 75

                #bottom_prior_con = torch.ones(n_parallel, 1, 2)
                #bottom_prior_con[..., 0] = 25
                #bottom_prior_con[..., 1] = 75

                #ee50_prior_mu, ee50_prior_sd = -50. * torch.ones(n_parallel, 1), 15. * torch.ones(n_parallel, 1)
                #slope_prior_mu, slope_prior_sd = -0.15 * torch.ones(n_parallel, 1), 0.1 * torch.ones(n_parallel, 1)

                #self.top_prior_con = top_prior_con
                #self.bottom_prior_con = bottom_prior_con
                #self.ee50_prior_mu = ee50_prior_mu
                #self.ee50_prior_sd = ee50_prior_sd
                #self.slope_prior_mu = slope_prior_mu
                #self.slope_prior_sd = slope_prior_sd
                
#################### NOT COMPLETE ####################

class PharmacoModel(ExperimentModel):
    def __init__(self, D_v=400., p=3, theta_loc=None, theta_covmat=None, epsilon_scale=None, nu_scale=None, n_parallel=1, obs_label="y"):
        super().__init__()
        self.D_v = D_v
        self.p = p

        #self.theta_mu = theta_mu if theta_mu is not None \
        #    else torch.zeros(n_parallel, 1, k, d)

        self.theta_loc = (theta_loc if theta_loc is not None else torch.tensor([1., 0.1, 20.]).log())
        self.theta_covmat = (theta_covmat if theta_covmat is not None else torch.eye(self.p) * 0.05)
        self.epsilon_scale = (epsilon_scale if epsilon_scale is not None else math.sqrt(0.01))
        self.nu_scale = (nu_scale if nu_scale is not None else math.sqrt(0.1))
        self.obs_label = obs_label
        self.n_parallel = n_parallel
        self.var_names = ["theta"]
        self.var_dim = 1
        self.sanity_check()

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                theta_shape = batch_shape + self.theta_loc.shape[-2:]
                print("design.shape", design.shape)
                print("self.theta_loc.shape[-2:]", self.theta_loc.shape[-2:])
                print("self.theta_loc.shape", self.theta_loc.shape)
                print("batch_shape", batch_shape)
                print("theta_shape", theta_shape)
                #theta = pyro.sample("theta", dist.MultivariateNormal(self.theta_loc.expand(batch_shape), self.theta_covmat.expand(batch_shape)).to_event(2)).exp()
                theta = pyro.sample("theta", dist.MultivariateNormal(self.theta_loc, self.theta_covmat).to_event(2)).exp()

                #print("theta", theta.shape)
                
                # unpack latents [these are exp-ed already!]
                k_a, k_e, V = [theta[..., [i]] for i in range(self.p)]
                assert (k_a > k_e).all()
                # compute concentration at time t=xi
                # shape of mean is [batch, n] where n is number of obs per design
                mean = ((self.D_v / V) * (k_a / (k_a - k_e)) * (
                        torch.exp(-torch.einsum("...ijk, ...ik->...ij", design, k_e))
                        - torch.exp(-torch.einsum("...ijk, ...ik->...ij", design, k_a))))
                sd = torch.sqrt((mean * self.epsilon_scale) ** 2 + self.nu_scale ** 2)
                emission_dist = dist.Normal(mean, sd).to_event(1)
                #print("emission_dist", emission_dist.shape)
                y = pyro.sample(self.obs_label, emission_dist)
                #print("y", y.shape)
                return y

        return model

    def reset(self, n_parallel, theta_loc=None, theta_covmat=None):
        self.n_parallel = n_parallel
        self.theta_loc = (theta_loc if theta_loc is not None else torch.tensor([1., 0.1, 20.]).log())
        self.theta_covmat = (theta_covmat if theta_covmat is not None else torch.eye(self.p) * 0.05)