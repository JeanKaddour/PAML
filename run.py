import argparse

import numpy as np
import tensorflow as tf

from utils.algorithm_utils import discretise_region, filter_candidates, acquire_task, add_new_task
from utils.evaluation import Evaluation
from utils.init_utils import init_experiments, init_logger, str2bool, init_args


def run_experiments(**exp_params) -> None:
    logger = init_logger(exp_params)

    trajectory_generator, training_task_descriptors, test_task_confs, controls, dataset, \
    session, model, meta_learner, lhs_tasks, test_observations = init_experiments(**exp_params)

    meta_learner.train_model()

    evaluation = None
    if exp_params["evaluation"]:
        evaluation = Evaluation(test_task_grid=test_task_confs,
                                meta_learner=meta_learner,
                                kwargs=exp_params,
                                test_observations=test_observations)

        evaluation.evaluation_on_test_tasks(dataset=dataset, test_tasks_params=test_task_confs, iteration=0,
                                            controls=controls)

    for iteration in range(exp_params["task_budget"]):
        latent_task_variables_mean, latent_task_variables_var = meta_learner.get_H_space_subset(end_task_id=
                                                                                                exp_params[
                                                                                                    "n_initial_training_envs"] + iteration)

        candidates = discretise_region(latent_task_variables_mean=latent_task_variables_mean,
                                       slack_min_values=exp_params["slack_min_intervals"],
                                       slack_max_values=exp_params["slack_max_intervals"],
                                       grid_resolution=exp_params["candidate_grid_size"])
        candidates = filter_candidates(latent_task_variables_mean=latent_task_variables_mean,
                                       task_configurations=training_task_descriptors,
                                       candidates=candidates,
                                       config_space=exp_params[
                                           "observed_configuration_space_interval"],
                                       verbose=exp_params["verbose"], GPModel=model,
                                       session=session)

        logger.info(f"Number of candidates: {candidates.shape}")
        selected_task_descriptor = acquire_task(iteration=iteration,
                                                latent_task_variables_mean=latent_task_variables_mean,
                                                latent_task_variables_var=latent_task_variables_var,
                                                discretised_latent_space_region=candidates,
                                                task_descriptors=training_task_descriptors,
                                                meta_learner=meta_learner,
                                                lhs_tasks=lhs_tasks, model=model,
                                                **exp_params)
        logger.info(f"Acquired task configuration: {selected_task_descriptor}")
        dataset.add_configuration(new_configuration=selected_task_descriptor)
        acquired_task_observations = trajectory_generator.observe_trajectories(
            task_configurations=selected_task_descriptor,
            controls=controls,
            dim_states=exp_params["dim_states"])[0]
        meta_learner, training_task_descriptors = add_new_task(iteration=iteration,
                                                               meta_learner=meta_learner,
                                                               acquired_task_observations=acquired_task_observations,
                                                               controls=controls,
                                                               training_task_descriptors=training_task_descriptors,
                                                               selected_task_descriptor=selected_task_descriptor,
                                                               **exp_params)
        meta_learner.train_model()

        if exp_params["evaluation"]:
            evaluation.evaluation_on_test_tasks(dataset=dataset, test_tasks_params=test_task_confs,
                                                iteration=(iteration + 1),
                                                controls=controls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", default=False, type=str2bool)
    parser.add_argument("--seed", default=1, type=int)

    # PAML parameters
    parser.add_argument("--task_budget", default=15, type=int)
    parser.add_argument("--n_initial_training_envs", default=3, type=int)
    parser.add_argument("--initial_training_configurations", default="LHS", type=str)
    parser.add_argument("--utility_function", default="PAML", type=str)
    parser.add_argument("--candidate_grid_size", default=100, type=int)

    # Environment / Dynamics parameters
    parser.add_argument("--env_name", default="cartpole", type=str)
    parser.add_argument("--policy", default="ALTERNATE", type=str)
    parser.add_argument("--control_signal_upper_bound", default=25., type=float)
    parser.add_argument("--alternations", default=10, type=int)
    parser.add_argument("--dt", default=.125, type=float)
    parser.add_argument("--training_trajectory_length", default=100, type=int)

    # Latent space parameters
    parser.add_argument("--dim_h", default=2, type=int)
    parser.add_argument("--slack_min_const_dim_1", default=-10., type=float)
    parser.add_argument("--slack_max_const_dim_1", default=10., type=float)
    parser.add_argument("--slack_min_const_dim_2", default=-10., type=float)
    parser.add_argument("--slack_max_const_dim_2", default=10., type=float)

    # Configuration space interval parameters
    parser.add_argument("--under_specified_system", default=False, type=str2bool)
    parser.add_argument("--over_specified_system", default=False, type=str2bool)
    parser.add_argument("--config_space_dim", default=2, type=int)
    parser.add_argument("--observed_config_space_dim", default=2, type=int)
    parser.add_argument("--config_interval_lower_bound_dim_1", default=.4, type=float)
    parser.add_argument("--config_interval_upper_bound_dim_1", default=3., type=float)
    parser.add_argument("--config_interval_lower_bound_dim_2", default=.4, type=float)
    parser.add_argument("--config_interval_upper_bound_dim_2", default=3., type=float)
    parser.add_argument("--config_interval_lower_bound_dim_3", default=.5, type=float)
    parser.add_argument("--config_interval_upper_bound_dim_3", default=5., type=float)
    parser.add_argument("--unobserved_parameter_lower_bound_dim_1", default=.4, type=float)
    parser.add_argument("--unobserved_parameter_upper_bound_dim_1", default=3., type=float)
    parser.add_argument("--config_space_decimals", default=2, type=int)

    # Evaluation parameters
    parser.add_argument("--evaluation", default=True, type=str2bool)
    parser.add_argument("--n_tasks_per_dim_of_evaluation_task_grid", default=10, type=int)
    parser.add_argument("--test_trajectory_length", default=100, type=int)
    parser.add_argument("--oracle", default=False, type=str2bool)

    # SVGP learning parameters
    parser.add_argument("--n_inducing_points", default=300, type=int)
    parser.add_argument("--data_normalization", default=True, type=str2bool)
    parser.add_argument("--training_steps", default=5000, type=int)
    parser.add_argument("--latent_variable_inference_steps", default=100, type=int)
    parser.add_argument("--learning_rate", default=1e-2, type=float)
    parser.add_argument("--batch_size", default=1000,
                        type=int)

    ARGS = parser.parse_args()
    args = vars(ARGS)

    init_args(args)

    tf.set_random_seed(args["seed"])
    np.random.seed(args["seed"])

    run_experiments(**args)
