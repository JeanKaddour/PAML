# Probabilistic Active Meta-Learning

In this work, we introduce **task selection based on prior experience** into a meta-learning algorithm by conceptualizing the learner and the active meta-learning setting using a probabilistic latent variable model.

[Link to the paper](https://arxiv.org/abs/2007.08949)

This repository implements the models and algorithms necessary to reproduce experiments (i)-(iii). 
To reproduce the results, you can run `batch.sh` (please do not change default values for certain parameters in `run.py`).

The core components of the repository are:

* `run.py`: script to run the PAML algorithm including all parameters
* `env`: directory for configuring and observing environments 
    * `controls.py`: generates control signals
    * `environment_configurator.py`: configures `dm_control` environments
    * `to.py`: observes trajectories of the environments given controls
* `models`: directory for the PAML model
    * `meta_learner.py`: trains the model and infers latent task variables
    * `mlgp.py`: the meta-learning (sparse variational) gaussian process model
    * `tp.py`: predicts trajectories (for evaluation)
* `utility_functions`: directory for the in the paper used utility functions and baselines
    * `paml.py`: PAML
    * `lhs.py`: Latin Hypercube Sampling
    * `uni.py`: Uniform sampling       
* `utils`: directory for miscellaneous tools
    * `algorithm_utils.py`: separated key steps of the PAML algorithm
    * `dataset.py`: stores and prepares trajectory observations
    * `evaluation.py`: evaluates the model's performance on test tasks
    
    
## Dependencies
This code was tested in ``Python 3.7``. 
The dependencies can be found in ``requirements.txt``.

## Installation 

1. Download and install [MuJoCo Pro 2.00](https://www.roboti.us/index.html)
    * You need a license and you can request a [trial license](https://www.roboti.us/license.html) for 30 days
    * At installation time, `dm_control`, looks for the MuJoCo headers in `~/.mujoco/mujoco200_$PLATFORM/include`
    * At runtime, `dm_control` looks for the MuJoCo license key file at `~/.mujoco/mjkey.txt`
2. Install all dependencies with `pip install -r requirements.txt`

## Usage examples

```
# Under-specified cart-pole environment
python3 run.py --env_name="cartpole" --utility_function="PAML" --seed=1  --under_specified_system True --observed_config_space_dim=1
 
# Fully-specified cart-pole environment
python3 run.py --env_name="cartpole" --utility_function="PAML" --seed=1  

# Fully-specified pendubot environment
python3 run.py --env_name="pendubot" --utility_function="PAML" --seed=1  

# Fully-specified cart-double-pole environment
python3 run.py --env_name="cartdoublepole" --utility_function="PAML" --seed=1  

# Over-specified cart-pole environment
python3 run.py --env_name="cartpole" --utility_function="PAML" --seed=1  --over_specified_system True --observed_config_space_dim=3 --config_space_dim=2

```

## Options for parameters

Parameters that require string values:

* `--env_name`: `'cartpole'`, `'cartdoublepole'`, `'pendubot'` 
* `--utility_function`: `'PAML'`, `'LHS'`, `'UNI'`
* `--policy`: `'ALTERNATE'`
* `--initial_training_configurations` : `'LHS'`, `'UNI'` 

Parameters that require boolean values:

* `--verbose`: printing additional information
* `--evaluation`: evaluation of the MLGP on a test task grid
* `--under_specified_system`: enables an unobserved, stochastic configuration dimension
* `--oracle` : initial training on the test task grid
* `--data_normalization` : normalization of training data over all dimensions

## Parameters intervals
The task paramater interval can be specified through the console, e.g., 

```
# By default, the following command runs an experiment with cart-pole tasks with pendulum mass in [0.5, 3.0] kg
python3 run.py --env_name="cartpole" --utility_function="PAML" --seed=1 --config_interval_lower_bound_dim_1=0.5 --config_interval_upper_bound_dim_1=3.0
```

In order to change the environment's parameterization (e.g., which configuration interval dimension corresponds to mass, length, radius, etc.), please have a look at `env/environment_configurator.py`

## Cite
```
@inproceedings{kaddour2020paml,
  title={Probabilistic Active-Meta Learning},
  author={Kaddour, Jean and Saemundsson, Steindor and Deisenroth, Marc Peter},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```