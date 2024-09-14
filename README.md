# Bayesian Experimental Design Through Reinforcement Learning

Note: This repository and its contents support the coursework of the INM363 module at City, University of London.

The code presented here is an updated version of the work found in the [RL-BOED](https://github.com/csiro-mlai/RL-BOED) repository and corresponding paper [Optimizing Sequential Experimental Design with Deep Reinforcement Learning](https://arxiv.org/abs/2202.00821). This version works with the latest versions of PyTorch, NumPy, and Gymnasium (note the difference from Gym). Since [Akro](https://github.com/rlworkgroup/akro) and [Garage](https://github.com/rlworkgroup/garage/) do not support the latest versions of these libraries (at least as of June 2024), they are imported and edited in this repository as separate folders. Note that we use the [2021.03 release](https://github.com/rlworkgroup/garage/tree/release-2021.03) of Garage since it was used in [RL-BOED](https://github.com/csiro-mlai/RL-BOED).

### Requirements
- Python 3.9+ - we use Python 3.9.13
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/) - we use PyTorch 2.3.0
- All other requirements listed in [**requirements.txt**](requirements.txt) - specific versions are listed

## Background

Optimal experimental design is the area dedicated to the optimal execution of experiments, with respect to some allocation of resources. Much work on optimal experimental design falls under the Bayesian setting, where we seek to reduce the uncertainty about our parameters of interest through experimentation. Conducting these experiments sequentially has recently brought about the use of reinforcement learning, where an agent is trained to navigate the design space to select the most informative designs for experimentation. However, there is still a lack of understanding about the benefits and drawbacks of using certain reinforcement learning algorithms to train these agents. In our work, we explore several reinforcement learning algorithms based on the state-of-the-art soft actor-critic method, and apply these to three Bayesian experimental design problems. We examine the amount of time needed to train these agents through each algorithm, and assess the generalisability of each agent to different but related experimental design setups. We draw insights on which algorithm generally performs best, and under what circumstances one may wish to use a particular algorithm.

## Data

Our trained agents, alongside their results at evaluation time, can be found [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EpDON-jNQRlElzC_crrOVd8BLTZeAa3YfN-BfrNehvAiCA?e=gRNk9z). Each folder beginning with 'boed_results' represents a set of 10 agents, each trained on a unique random seed. The agents are trained with a specific algorithm under a certain set of hyperparameters, and on a particular Bayesian experimental design problem. Folders with 'ces' are agents trained on the 'Constant Elasticity of Substitution' experiment, 'location' is the 'Location Finding' experiment, and 'docking' is the 'Biomolecular Docking' experiment.

## Algorithms

We explore [Randomised Ensembled Double Q-Learning (REDQ)](https://arxiv.org/abs/2101.05982), [Dropout Q-Functions for Doubly Efficient Reinforcement Learning (DroQ)](https://arxiv.org/abs/2110.02034), [Sample-Efficient Reinforcement Learning by Breaking the Replay Ratio Barrier (SBR)](https://openreview.net/forum?id=OpC-9aBBVJe), and [A Simple Unified Framework for Ensemble Learning in Deep Reinforcement Learning (SUNRISE)](https://arxiv.org/abs/2007.04938). These all extend the [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905) algorithm. [Proximal Policy Optimisation (PPO)](https://arxiv.org/abs/1707.06347) and [Trust Region Policy Optimisation (TRPO)](https://arxiv.org/abs/1502.05477) are two other algorithms that can be used.

## Bayesian Experimental Design Problems

### Location Finding

There are $K$ objects on a $d$-dimensional space, and in this experiment we need to identify their locations $\boldsymbol{\theta} = \{\theta_{i}\}_{i = 1}^{K}$. Each object emits a signal, which obeys the inverse-square law. We need to select different designs $\xi$, which are essentially locations (or points) on the $d$-dimensional space. The signal strength increases as we select $\xi$ closer to these $K$ objects, and it decays as we choose $\xi$ further away from these objects.

The total intensity at point $\xi$ is the superposition of the individual intensities for each object, $$\mu(\boldsymbol{\theta}, \xi) = b + \sum_{i = 1}^{K} \frac{\alpha}{m + ||\theta_{i} - \xi||^{2}},$$ where $\alpha$ is a constant, $b > 0$ is a constant controlling the background signal, and $m > 0$ is a constant controlling the maximum signal. The total intensity is then used in the likelihood function calculation.

For object $\theta_{i} \in \mathbb{R}^{d}$, its prior is given by $$\theta_{i} \sim \mathcal{N}_{d}(\boldsymbol{0}, I),$$ where $\boldsymbol{0}$ is the mean vector, and $I$ is the covariance matrix, an identity matrix, both with dimension $d$. For a given design $\xi$, the likelihood function is given by $$y \mid \boldsymbol{\theta}, \xi \sim \mathcal{N}(\log \mu(\boldsymbol{\theta}, \xi), \sigma^2).$$

| Parameter  | Value |
| ------------- | ------------- |
| $K$ | 2 |
| $d$ | 2 |
| $\alpha$ | 1 |
| $b$ | 0.1 |
| $m$ | 0.0001 |
| $\sigma$ | 0.5 |

### Constant Elasticity of Substitution

We have two baskets $\boldsymbol{x}, \boldsymbol{x'} \in [0, 100]^{3}$ of goods, and a human indicates their preference of the two baskets on a sliding 0-1 scale. For example, one basket could contain two chocolate bars, while the other basket contains an apple. The exact items in the baskets are indicated numerically and are not exactly real-world items, at least in our model. The CES model \citep{arrowchen} with latent variables $(\rho, \boldsymbol{\alpha}, u)$, which all characterise the human’s utility or preferences for the different items, is then used to measure the difference in utility of the baskets. The goal is to design the baskets in a way that allows for inference of the latent variable values, otherwise reducing our uncertainty about them. Both baskets are 3-tuples, meaning that we have $3 + 3 = 6$ design space dimensions $(\xi = (\boldsymbol{x}, \boldsymbol{x'}))$.

The CES model \citep{arrowchen} defines the utility $U(\boldsymbol{x})$ for a basket of goods $\boldsymbol{x}$ as, $$U(\boldsymbol{x}) = \left( \sum_{i} x_i^{\rho} \alpha_{i} \right)^{\frac{1}{\rho}},$$ where $\rho$ and $\boldsymbol{\alpha}$ are latent variables defined with the prior distributions explained below. This utility function, which is a measure of satisfaction in economic terms, is then used in the likelihood function calculation.

We use the following priors for $(\rho, \boldsymbol{\alpha}, u)$:
$$\rho \sim \text{Beta}(1, 1),
\boldsymbol{\alpha} \sim \text{Dirichlet}([1, 1, 1]),
u \sim \text{Log-Normal}(1, 3^{2}).$$

The likelihood function is the preference of the human on a sliding 0-1 scale, which is based on $U(\boldsymbol{x}) - U(\boldsymbol{x'})$. For a given design $\xi$, the likelihood function is given by, $$\mu_{\eta} = u \cdot (U(\boldsymbol{x}) - U(\boldsymbol{x'})) \\
\sigma_{\eta} = \nu u \cdot (1 + \|\boldsymbol{x} - \boldsymbol{x'}\|) \\
\eta \sim \mathcal{N}(\mu_{\eta}, \sigma_{\eta}^{2}) \\
y = \text{clip}(s(\eta), \epsilon, 1 - \epsilon),$$
where $\nu = 0.005$, $\epsilon = 2^{-22}$, and $s(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function. Notice that the normally distributed $\eta$ is passed through the sigmoid function, bounding $\eta \in [0, 1]$. Censoring/clipping is applied, which limits the distribution by setting values above $u = 1 - \epsilon$ to be equal to $1 - \epsilon$, and values below $l = \epsilon$ to be equal to $\epsilon$.

### Biomolecular Docking

## Code

### File Structure

### Training the Agents

### Evaluating the Agents
