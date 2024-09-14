# Bayesian Experimental Design Through Reinforcement Learning

Note: This repository and its contents support the coursework of the INM363 module at City, University of London.

The code presented here is an updated version of the work found in the [RL-BOED](https://github.com/csiro-mlai/RL-BOED) repository and corresponding paper [Optimizing Sequential Experimental Design with Deep Reinforcement Learning](https://arxiv.org/abs/2202.00821). This version works with the latest versions of PyTorch, NumPy, and Gymnasium (note the difference from Gym). Since [Akro](https://github.com/rlworkgroup/akro) and [Garage](https://github.com/rlworkgroup/garage/) do not support the latest versions of these libraries (at least as of June 2024), they are imported and edited in this repository as separate folders. Note that we use the [2021.03 release](https://github.com/rlworkgroup/garage/tree/release-2021.03) of Garage since it was used in [RL-BOED](https://github.com/csiro-mlai/RL-BOED).

### Requirements
- Python 3.9 - we use Python 3.9.13
- [PyTorch (with CUDA for GPU usage)](https://pytorch.org/get-started/locally/) - we use PyTorch 2.3.0
- All other requirements listed in [**requirements.txt**](requirements.txt) - specific versions are listed

## Background

Optimal experimental design is the area dedicated to the optimal execution of experiments, with respect to some allocation of resources. Much work on optimal experimental design falls under the Bayesian setting, where we seek to reduce the uncertainty about our parameters of interest through experimentation. Conducting these experiments sequentially has recently brought about the use of reinforcement learning, where an agent is trained to navigate the design space to select the most informative designs for experimentation. However, there is still a lack of understanding about the benefits and drawbacks of using certain reinforcement learning algorithms to train these agents. In our work, we explore several reinforcement learning algorithms based on the state-of-the-art soft actor-critic method, and apply these to three Bayesian experimental design problems. We examine the amount of time needed to train these agents through each algorithm, and assess the generalisability of each agent to different but related experimental design setups. We draw insights on which algorithm generally performs best, and under what circumstances one may wish to use a particular algorithm.

## Data

Our trained agents, alongside their results at evaluation time, can be found [here](https://cityuni-my.sharepoint.com/:f:/g/personal/yasir-zubayr_barlas_city_ac_uk/EpDON-jNQRlElzC_crrOVd8BLTZeAa3YfN-BfrNehvAiCA?e=gRNk9z). Each folder beginning with 'boed_results' represents a set of 10 agents, each trained on a unique random seed. The agents are trained with a specific algorithm under a certain set of hyperparameters, and on a particular Bayesian experimental design problem. Folders with 'ces' are agents trained on the 'Constant Elasticity of Substitution' experiment, 'location' is the 'Location Finding' experiment, and 'docking' is the 'Biomolecular Docking' experiment.

## Bayesian Experimental Design Problems

### Location Finding

### Constant Elasticity of Substitution

### Biomolecular Docking

## Code

### File Structure

### Training the Agents

### Evaluating the Agents
