import os
import argparse
import itertools
import logging

import numpy as np
import tensorflow as tf
from gpflow import kernels

from models.gpflowmod import likelihoods
from models.mlgp import MLSVGP

from models.tp import TrajectoryPredictor
from models.meta_learner import MetaLearner
from utils.math_utils import sample_from_multidim_interval_uniformly
from env.to import TrajectoryObserver
from env.controls import generate_control_signals
from utils.dataset import MultiEnvData
from utility_functions.lhs import latin_hypercube_sampling_maxi_min


def init_experiments(**args):
    logger = logging.getLogger("paml")

    trajectory_observer = TrajectoryObserver(args["env_name"], args["seed"], args["dt"], args["under_specified_system"],
                                             args["over_specified_system"],
                                             args["unobserved_parameter_bounds"])
    test_tasks_configurations = create_configurations_grid(
        configuration_space_interval=args["configuration_space_interval"],
        grid_resolution=args[
            "n_tasks_per_dim_of_evaluation_task_grid"])
    # Finding training tasks
    training_tasks_configurations = test_tasks_configurations if args[
        "oracle"] else create_training_task_configuration_array(**args)

    lhs_tasks = np.round(training_tasks_configurations[args["n_initial_training_envs"]:],
                         args["config_space_decimals"]) if args[
                                                               "utility_function"] == "LHS" else None
    training_tasks_configurations = training_tasks_configurations[:args["n_initial_training_envs"]]

    if args["config_space_dim"] < args["observed_config_space_dim"]:
        test_tasks_configurations = np.hstack((test_tasks_configurations, np.ones(
            (test_tasks_configurations.shape[0], args["observed_config_space_dim"] - args["config_space_dim"]))))

    controls = generate_control_signals(**args)

    logger.info("Observe initial training tasks...")
    training_observations = trajectory_observer.observe_trajectories(task_configurations=training_tasks_configurations,
                                                                     controls=controls,
                                                                     dim_states=args["dim_states"])

    logger.info("Observe test tasks...")
    test_observations = training_observations if args[
        "oracle"] else trajectory_observer.observe_trajectories(
        task_configurations=test_tasks_configurations,
        controls=controls,
        dim_states=args["dim_states"])

    dataset = MultiEnvData(env_name=args["env_name"], observations=training_observations, controls=controls,
                           task_configurations=training_tasks_configurations,
                           trajectory_length=args["training_trajectory_length"],
                           data_normalization=args["data_normalization"])

    model = create_meta_learning_model(**args)
    trajectory_predictor = create_trajectory_predictor(model=model, **args)

    session = tf.Session()
    meta_learner = MetaLearner(model=model, trajectory_predictor=trajectory_predictor, session=session, dataset=dataset,
                               **args)

    return trajectory_observer, training_tasks_configurations, test_tasks_configurations, controls, dataset, session, \
           model, meta_learner, lhs_tasks, test_observations


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def create_trajectory_predictor(model, **kwargs):
    tp = TrajectoryPredictor(
        model=model,
        dim_states=kwargs["dim_states"], dim_actions=kwargs["dim_actions"],
        dim_angles=kwargs["dim_angles"], trajectory_length=kwargs["test_trajectory_length"],
        dim_h=kwargs["dim_h"],
        inducing_points=kwargs["n_inducing_points"],
    )
    return tp


def create_meta_learning_model(**kwargs):
    Z = np.random.randn(kwargs["n_inducing_points"], kwargs["dim_in"] + kwargs["dim_h"])
    mean_func = None
    kernel = kernels.RBF(kwargs["dim_in"] + kwargs["dim_h"], ARD=True)
    likelihood = likelihoods.MultiGaussian(dim=kwargs["dim_out"])
    latent_to_conf_space_kernel = kernels.RBF(kwargs["dim_h"], ARD=True)
    latent_to_conf_space_likelihood = likelihoods.Gaussian()

    model = MLSVGP(
        dim_in=kwargs["dim_in"], dim_out=kwargs["dim_out"],
        dim_h=kwargs["dim_h"], num_h=kwargs["n_envs"],
        kern=kernel, likelihood=likelihood, mean_function=mean_func,
        Z=Z,
        observed_config_space_dim=kwargs["observed_config_space_dim"],
        latent_to_conf_space_kernel=latent_to_conf_space_kernel,
        latent_to_conf_space_likelihood=latent_to_conf_space_likelihood)

    return model


def init_logger(args):
    logger = logging.getLogger('paml')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(args['experiment_path'] + '/logs.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.debug(args)
    tf_logger = logging.getLogger("tensorflow")
    tf_logger.setLevel(logging.DEBUG)
    tf_logger.addHandler(fh)
    tf_logger.addHandler(ch)
    return logger


def create_training_task_configuration_array(**args):
    training_tasks_configurations = None
    if args["initial_training_configurations"] == "LHS":
        training_tasks_configurations = latin_hypercube_sampling_maxi_min(
            args["n_initial_training_envs"] + args["task_budget"],
            args["observed_config_space_dim"], **args)
    elif args["initial_training_configurations"] == "UNI":
        training_tasks_configurations = sample_from_multidim_interval_uniformly(
            args["observed_configuration_space_interval"], args["n_initial_training_envs"])
        training_tasks_configurations = np.round(training_tasks_configurations, args["config_space_decimals"])

    if args["verbose"]:
        print(f"Sampled initial training tasks: {training_tasks_configurations}")

    return training_tasks_configurations


def create_endpoints_array(bounds: np.array):
    return np.array([item for item in itertools.product(*bounds)]).reshape(-1, bounds.shape[0])


def create_configurations_grid(configuration_space_interval, grid_resolution: int = 4, config_space_decimals: int = 2):
    grids = []
    for dim in configuration_space_interval:
        grid = np.linspace(dim[0], dim[1], grid_resolution)
        grid = np.round(grid, config_space_decimals)
        grids.append(grid)
    mesh = np.meshgrid(*grids)
    configurations = []
    for dim in mesh:
        transformed_dim = np.transpose(dim).flatten()
        configurations.append(transformed_dim)
    configurations = np.transpose(configurations)
    return configurations.reshape(-1, len(configuration_space_interval))


def init_args(args):
    if args["oracle"]:
        args["n_initial_training_envs"] = args[
                                              "n_tasks_per_dim_of_evaluation_task_grid"] ** args[
                                              "config_space_dim"]
        args["n_active_tasks"] = args["n_initial_training_envs"]

        args["n_envs"] = args["n_active_tasks"]

    else:
        args["n_envs"] = args["n_initial_training_envs"] + args["task_budget"] + args[
            "n_tasks_per_dim_of_evaluation_task_grid"] ** args[
                             "config_space_dim"]

        args["n_active_tasks"] = args["n_initial_training_envs"]

    if args["env_name"] == "cartpole":
        args["dim_in"] = 6
        args["dim_out"] = 4
        args["dim_states"] = 4
        args["dim_actions"] = 1
        args["dim_angles"] = 1
    elif args["env_name"] == "cartdoublepole":
        args["dim_in"] = 9
        args["dim_out"] = 6
        args["dim_states"] = 6
        args["dim_actions"] = 1
        args["dim_angles"] = 2
    elif args["env_name"] == "pendubot":
        args["dim_in"] = 7
        args["dim_out"] = 4
        args["dim_states"] = 4
        args["dim_actions"] = 1
        args["dim_angles"] = 2

    args["observed_configuration_space_interval"] = []
    for dim in range(1, args["observed_config_space_dim"] + 1):
        args["observed_configuration_space_interval"].append(
            [args[f"config_interval_lower_bound_dim_{dim}"], args[f"config_interval_upper_bound_dim_{dim}"]])
    args["observed_configuration_space_interval"] = np.array(args["observed_configuration_space_interval"])

    args["configuration_space_interval"] = []
    for dim in range(1, args["config_space_dim"] + 1):
        args["configuration_space_interval"].append(
            [args[f"config_interval_lower_bound_dim_{dim}"], args[f"config_interval_upper_bound_dim_{dim}"]])
    args["configuration_space_interval"] = np.array(args["configuration_space_interval"])

    args["unobserved_parameter_bounds"] = [args['unobserved_parameter_lower_bound_dim_1'],
                                           args['unobserved_parameter_upper_bound_dim_1']]

    if args["dim_h"] == 1:
        args["slack_min_intervals"] = np.array(
            [args["slack_min_const_dim_1"]])
        args["slack_max_intervals"] = np.array(
            [args["slack_max_const_dim_1"]])
    elif args["dim_h"] == 2:
        args["slack_min_intervals"] = np.array(
            [args["slack_min_const_dim_1"], args["slack_min_const_dim_2"]])
        args["slack_max_intervals"] = np.array(
            [args["slack_max_const_dim_1"], args["slack_max_const_dim_2"]])

    folder_keys = [
        "env_name",
        "utility_function",
        "training_steps",
        "n_inducing_points",
        "seed"]

    experiment_path = "experiments/"
    model_path = "model/"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for key in folder_keys:
        experiment_path += f"{args[key]}-"
    experiment_path = experiment_path[:-1]

    if args["oracle"]:
        experiment_path += "-ORACLE"

    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        os.makedirs(experiment_path + "/model")
        os.makedirs(experiment_path + "/img")
        os.makedirs(experiment_path + "/img/rewards")
        os.makedirs(experiment_path + "/img/H_space")
        os.makedirs(experiment_path + "/img/H_space/live_inference")

    checkpoint_path = experiment_path + "/model.ckpt"
    args["experiment_path"] = experiment_path
    args["checkpoint_path"] = checkpoint_path
    args["model_path"] = model_path
