import numpy as np
from utils.math_utils import mu_std


class MultiEnvData:

    def __init__(self, env_name, observations=None, controls=None, task_configurations=None, trajectory_length=50,
                 data_normalization=False):
        self.env_name = env_name
        self.trajectories = {}
        self.rewards = {}
        self.data = {}
        self.num_envs = 0
        self.trajectory_length = trajectory_length
        self.data_normalization = data_normalization
        self.task_configurations = task_configurations

        if observations is not None and controls is not None:
            for i in range(len(observations)):
                self.add_observations(observations[i], controls, i)

    def add_configuration(self, new_configuration):
        self.task_configurations = np.vstack((self.task_configurations, new_configuration))

    def add_observations(self, states, controls, id):
        trajectory_data = {"states": states, "controls": controls}

        if id in self.trajectories:
            self.trajectories[id].append(trajectory_data)
        else:
            self.trajectories[id] = [trajectory_data]
            self.num_envs += 1

    def get_inputs_outputs(self, states, controls):

        states_transformed = self.state_transform(states)
        inputs = np.hstack([states_transformed[:-1], controls[:-1]])
        outputs = states[1:] - states[:-1]
        return inputs, outputs

    def prepare_data(self):
        all_inputs_trajectory = []
        all_outputs_trajectory = []
        all_ids_trajectory = []
        n_trajectories = 0
        for id in self.trajectories:
            trajectories = self.trajectories[id]
            for trajectory in trajectories:
                states = trajectory["states"]
                controls = trajectory["controls"]
                inputs, outputs = self.get_inputs_outputs(states, controls)
                ids = np.int32(inputs.shape[0] * [id])

                all_inputs_trajectory.append(inputs)
                all_outputs_trajectory.append(outputs)
                all_ids_trajectory.append(ids.reshape(-1, 1))
                n_trajectories += 1

        all_inputs = np.vstack(all_inputs_trajectory)
        all_outputs = np.vstack(all_outputs_trajectory)

        inp_mu, inp_std = mu_std(all_inputs)
        out_mu, out_std = mu_std(all_outputs)
        if self.data_normalization:
            norm = lambda inp, mu, std: (inp - mu) / std
            inp_norm = [norm(inp, inp_mu, inp_std) for inp in all_inputs_trajectory]
            out_norm = [norm(out, out_mu, out_std) for out in all_outputs_trajectory]
        else:
            inp_norm = all_inputs_trajectory
            out_norm = all_outputs_trajectory

        n_data = all_inputs.shape[0]

        self.data["n_data"] = n_data
        self.data["n_trajectories"] = n_trajectories
        self.data["inputs"] = inp_norm
        self.data["outputs"] = out_norm
        self.data["inputs_unnormalised"] = all_inputs
        self.data["outputs_unnormalised"] = all_outputs
        self.data["ids"] = all_ids_trajectory
        self.data["inp_mu"] = inp_mu
        self.data["inp_std"] = inp_std
        self.data["out_mu"] = out_mu
        self.data["out_std"] = out_std

    def get_seq_batch(self, seq, si, ei):
        inp_seq = self.data["inputs"]
        out_seq = self.data["outputs"]
        ids_seq = self.data["ids"]
        inp_seq = [inp_seq[i] for i in seq]
        out_seq = [out_seq[i] for i in seq]
        ids_seq = [ids_seq[i] for i in seq]
        D = inp_seq[0].shape[1]
        E = out_seq[0].shape[1]

        X_b = np.vstack(inp_seq[si:ei]).reshape(-1, D)
        Y_b = np.vstack(out_seq[si:ei]).reshape(-1, E)
        ids_b = np.vstack(ids_seq[si:ei]).reshape(-1)
        ids_unique = np.unique(ids_b)
        return X_b, Y_b, ids_b, ids_unique, self.task_configurations

    def state_transform(self, states):
        states_transformed = states

        if self.env_name == "cartpole":
            states_transformed = np.zeros((states.shape[0], states.shape[1] + 1))
            states_transformed[:, 0] = np.cos(states[:, 0])
            states_transformed[:, 1] = np.sin(states[:, 0])
            states_transformed[:, 2] = states[:, 1]
            states_transformed[:, 3] = states[:, 2]
            states_transformed[:, 4] = states[:, 3]

        elif self.env_name == "cartdoublepole":
            states_transformed = np.zeros((states.shape[0], states.shape[1] + 2))
            states_transformed[:, 0] = np.cos(states[:, 0])
            states_transformed[:, 1] = np.sin(states[:, 0])
            states_transformed[:, 2] = np.cos(states[:, 1])
            states_transformed[:, 3] = np.sin(states[:, 1])
            states_transformed[:, 4] = states[:, 2]
            states_transformed[:, 5] = states[:, 3]
            states_transformed[:, 6] = states[:, 4]
            states_transformed[:, 7] = states[:, 5]

        elif self.env_name == "pendubot":
            states_transformed = np.zeros((states.shape[0], states.shape[1] + 2))
            states_transformed[:, 0] = np.sin(states[:, 0])
            states_transformed[:, 1] = np.cos(states[:, 0])
            states_transformed[:, 2] = np.sin(states[:, 1])
            states_transformed[:, 3] = np.cos(states[:, 1])
            states_transformed[:, 4] = states[:, 2]
            states_transformed[:, 5] = states[:, 3]
        return states_transformed

    def get_shortened_inputs(self, steps_per_trajectory):
        self.prepare_data()
        trajectory_inputs = []
        for id in self.trajectories:
            trajectories = self.trajectories[id]
            for trajectory in trajectories:
                states = trajectory["states"]
                controls = trajectory["controls"]
                inputs, _ = self.get_inputs_outputs(states, controls)
                trajectory_inputs.append(inputs)

        trajectory_inputs = np.array(trajectory_inputs)
        trajectory_tmp = []
        for i in range(trajectory_inputs.shape[0]):
            for j in range(steps_per_trajectory):
                trajectory_tmp.append(trajectory_inputs[i, j])
        trajectory_shortened = np.vstack(trajectory_tmp)
        return trajectory_shortened
