import numpy as np
from utility_functions.paml import select_new_latent_point_PAML
from utility_functions.uni import acquire_random_task_descriptor

import logging


def add_new_task(iteration, meta_learner, acquired_task_observations, controls, training_task_descriptors,
                 selected_task_descriptor, **args):
    meta_learner.dataset.add_observations(states=acquired_task_observations, controls=controls,
                                          id=args["n_initial_training_envs"] + iteration)
    training_task_descriptors = np.vstack((training_task_descriptors, selected_task_descriptor))
    meta_learner.n_active_tasks = len(training_task_descriptors)
    return meta_learner, training_task_descriptors


def generate_input_candidates(latent_points, inputs, length, dim_in, dim_h):
    H_start_idx = dim_in
    H_end_idx = dim_in + dim_h
    input_IDs = np.linspace(start=0, stop=len(inputs), num=length, endpoint=False, dtype=np.int)
    selected_inputs = inputs[input_IDs]
    candidates = []
    for latent_point in latent_points:
        new_candidate = np.hstack((selected_inputs, length * [latent_point]))
        candidates.append(new_candidate)
    return np.array(candidates).reshape(-1, 1), [H_start_idx, H_end_idx]


def acquire_task(iteration, latent_task_variables_mean, latent_task_variables_var, discretised_latent_space_region,
                 task_descriptors, meta_learner, lhs_tasks, model, **args):
    selected_task_descriptor = None
    if args["utility_function"] == "LHS":
        selected_task_descriptor = lhs_tasks[iteration].reshape(1, -1)
    elif args["utility_function"] == "UNI":
        selected_task_descriptor = np.round(
            acquire_random_task_descriptor(args['observed_configuration_space_interval']),
            args['config_space_decimals'])
    elif args["utility_function"] == "PAML":
        selected_latent_point = select_new_latent_point_PAML(
            np.hstack((latent_task_variables_mean, latent_task_variables_var)), discretised_latent_space_region)

        pred_task_descriptor_mean, pred_task_descriptor_var = latent_to_config_space_regressor(
            latent_task_variables_mean=latent_task_variables_mean,
            task_descriptors=task_descriptors,
            latent_candidate=selected_latent_point,
            verbose=args['verbose'], GPModel=model,
            session=meta_learner.session)
        selected_task_descriptor = np.round(pred_task_descriptor_mean,
                                            args["config_space_decimals"])
        logger.info(f"Selected latent point {selected_latent_point}")
    return selected_task_descriptor


def discretise_region(latent_task_variables_mean, slack_min_values, slack_max_values, grid_resolution=3,
                      decimals=5):
    min_values = np.min(latent_task_variables_mean, 0)
    max_values = np.max(latent_task_variables_mean, 0)
    grids = []
    for dim in range(latent_task_variables_mean.shape[1]):
        min_values[dim] += slack_min_values[dim]
        max_values[dim] += slack_max_values[dim]
        grid = np.round(np.linspace(min_values[dim], max_values[dim], grid_resolution), decimals)
        grids.append(grid)
    mesh = np.meshgrid(*grids)
    final = []
    for dim in mesh:
        transformed_dim = np.transpose(dim).flatten()
        final.append(transformed_dim)
    final = np.transpose(final)
    return final.reshape(-1, latent_task_variables_mean.shape[1])


def filter_candidates(latent_task_variables_mean, task_configurations, candidates, config_space, verbose=False,
                      GPModel=None, session=None):
    new_latent_space = []
    pred_means, pred_vars = latent_to_config_space_regressor(latent_task_variables_mean, task_configurations,
                                                             candidates,
                                                             verbose=verbose, GPModel=GPModel, session=session)
    reject = False
    for id, conf in enumerate(pred_means):
        for i in range(conf.shape[0]):
            reject = False
            if conf[i] < config_space[i, 0] or conf[i] > config_space[i, 1]:
                reject = True
                break
        if not reject:
            new_latent_space.append(candidates[id])

    return np.array(new_latent_space)


def latent_to_config_space_regressor(latent_task_variables_mean, task_descriptors, latent_candidate,
                                     verbose=False, GPModel=None, session=None):
    if latent_candidate is None:
        return [], []
    elif latent_candidate.shape[0] == 0:
        return [], []

    mean, var = GPModel.build_predict_psi(latent_candidate)
    mean = mean.eval(session=session, feed_dict={GPModel.psi_ph: task_descriptors,
                                                 GPModel.H_unique_ph: np.arange(len(task_descriptors))})
    var = var.eval(session=session, feed_dict={GPModel.psi_ph: task_descriptors,
                                               GPModel.H_unique_ph: np.arange(len(task_descriptors))})
    if verbose:
        logger = logging.getLogger("paml")
        logger.info("Training latent points: ", latent_task_variables_mean, "\n")
        logger.info("Training task descriptors: ", task_descriptors, "\n")
        logger.info("Input latent point: ", latent_candidate, "\n")
        logger.info("Predicted task descriptor: ", mean)
        logger.info("Variance of predicted task descriptor: ", var)
    return mean, var
