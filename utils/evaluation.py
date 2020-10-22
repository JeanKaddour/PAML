import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm
import logging


class Evaluation:

    def __init__(self, test_task_grid, meta_learner,
                 kwargs, test_observations=None):
        self.test_task_grid = test_task_grid

        self.kwargs = kwargs
        self.data_normalization = kwargs["data_normalization"]

        self.test_observations = test_observations
        self.meta_learner = meta_learner
        self.oracle = self.kwargs["oracle"]
        self.logger = logging.getLogger("paml")

    def evaluation_on_test_tasks(self, dataset, test_tasks_params, iteration, controls):

        if self.oracle:

            task_start_index = 0
            num_test_tasks = test_tasks_params.shape[0]

        else:
            self.meta_learner.n_active_tasks += len(test_tasks_params)

            task_start_index = self.kwargs["n_initial_training_envs"] + self.kwargs["task_budget"]
            num_test_tasks = test_tasks_params.shape[0]

        X_test = np.zeros((num_test_tasks, controls.shape[0] - 1, self.kwargs["dim_in"]))
        Y_test = np.zeros((num_test_tasks, controls.shape[0] - 1, self.kwargs["dim_out"]))

        trajectory_length = X_test.shape[1]

        for i in range(num_test_tasks):
            x, y = dataset.get_inputs_outputs(self.test_observations[i], controls)
            if self.data_normalization:
                x = (x - dataset.data["inp_mu"]) / dataset.data["inp_std"]
                y = (y - dataset.data["out_mu"]) / dataset.data["out_std"]
            X_test[i] = x
            Y_test[i] = y

        nll = []
        rmse_dim_avg = []
        rmse_all_dim_list = []

        # Infer task variables for test tasks
        print("Inferring task variables for test tasks...")
        for j in tqdm(range(num_test_tasks)):
            self.meta_learner.infer_task_variable(task_start_index + j, self.test_observations[j], controls,
                                                  test_tasks_params[j,
                                                  :self.kwargs["observed_config_space_dim"]].reshape(1, -1))

        # Gather latent embeddings from session
        H = self.meta_learner.model.H.read_value(session=self.meta_learner.session)
        h_mu = H[:, :self.meta_learner.model.dim_h]

        h_var = np.exp(H[:, self.meta_learner.model.dim_h:])

        Luu = self.meta_learner.session.run(self.meta_learner.model.compute_Luu())

        XH_mu_test = np.zeros((num_test_tasks, trajectory_length, X_test.shape[2] + h_mu.shape[1]))
        for i in range(num_test_tasks):
            XH_mu_test[i] = np.hstack((X_test[i], np.array(trajectory_length * [h_mu[task_start_index + i]])))
        XH_var_train = np.zeros(
            (num_test_tasks, trajectory_length, X_test.shape[2] + h_var.shape[1], X_test.shape[2] + h_var.shape[1]))

        print("Computing predictions of test tasks...")
        for i in tqdm(range(num_test_tasks)):
            for j in range(trajectory_length):
                # for each task and trajectory step build block diagonal matrix with dimensions D x D
                XH_var_train[i, j] = block_diag(XH_var_train[i, j, : X_test.shape[2], : X_test.shape[2]],
                                                np.diag(h_var[task_start_index + i]))

            # Input dimensions: T x D and T x D x D
            fmean, fvar = self.meta_learner.predict_state(XH_mu_test[i], XH_var_train[i], Luu)

            # Get variational expectations.
            nll.append(-1. *
                       self.meta_learner.session.run(
                           self.meta_learner.model.likelihood.variational_expectations(fmean, fvar,
                                                                                       Y_test[i])))

            rmse_avg, rmse_all_dim = root_mean_squared_error(Y_test[i], fmean)
            rmse_all_dim_list.append(rmse_all_dim)
            rmse_dim_avg.append(rmse_avg)

            # Re-normalise outputs
            if self.data_normalization:
                Y_test[i] = (Y_test[i] * dataset.data["out_std"]) + dataset.data["out_mu"]

                for j in range(trajectory_length):
                    fmean[j] = (fmean[j] * dataset.data["out_std"]) + dataset.data["out_mu"]
                    fvar[j] = fvar[j].T @ fvar[j]
                    fvar[j] = fvar[j] * dataset.data["out_std"]

        text_file = open(self.kwargs["experiment_path"] + "/eval.txt", "a+")
        text_file.write("Iteration: " + str(iteration) + "\n")

        nll = np.sum(np.sum(np.array(nll), axis=1), axis=1)
        nll_file_path = self.kwargs["experiment_path"] + f"/nll={str(iteration)}.npz"
        np.savez(nll_file_path,
                 np.array(nll))

        rmse_dim_avg = np.array(rmse_dim_avg)
        rmse_dim_avg_file_path = self.kwargs["experiment_path"] + f"/rmse={str(iteration)}.npz"
        np.savez(rmse_dim_avg_file_path,
                 np.array(rmse_dim_avg))

        nll_sum = np.sum(nll)
        rmse_sum = np.sum(rmse_dim_avg)
        self.logger.info(f"NLL Sum: {nll_sum}")
        self.logger.info(f"RMSE Sum: {rmse_sum}")
        text_file.write(f"NLL Sum: {nll_sum} \n")
        text_file.write(f"RMSE Sum: {rmse_sum} \n")
        text_file.close()

        return nll_sum, rmse_sum


def root_mean_squared_error(y_true, y_pred,
                            sample_weight=None):
    output_errors = np.sqrt(np.average((y_true - y_pred) ** 2, axis=0,
                                       weights=sample_weight))
    return np.sum(output_errors), output_errors
