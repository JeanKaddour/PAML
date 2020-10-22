import tensorflow as tf
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger("paml")


class MetaLearner:

    def __init__(self, model, trajectory_predictor, dataset, session, training_saving_interval=100,
                 **kwargs):
        self.kwargs = kwargs
        self.rng = np.random.RandomState(kwargs["seed"])
        self.session = session
        self.model = model
        self.dataset = dataset
        self._build_model_graph()
        self.n_iterations = 0
        self.n_active_tasks = kwargs["n_active_tasks"]
        self.dim_h = kwargs["dim_h"]

        self.trajectory_predictor = trajectory_predictor

        self.fmean, self.fvar = self.trajectory_predictor.predict_state()

        self.training_saving_interval = training_saving_interval
        self.max_steps_to_keep = kwargs["training_steps"] / self.training_saving_interval + 1
        self.saver = tf.train.Saver(max_to_keep=int(self.max_steps_to_keep))

    def _build_model_graph(self):
        self.model_objective = -self.model.build_likelihood()
        self.model_train_step, self.model_infer_step, self.model_optimizer = \
            init_model(model=self.model, objective=self.model_objective,
                       session=self.session, learning_rate=self.kwargs["learning_rate"])

    def _set_inducing(self):

        n_data = self.dataset.data["n_data"]
        n_inducing_points = self.kwargs["n_inducing_points"]
        X = np.vstack(self.dataset.data["inputs"])
        Z = self.model.feature.Z.read_value(session=self.session)

        diff = n_inducing_points - n_data
        if diff >= 0:
            Z[:n_data, :X.shape[1]] = X
        else:
            seq = np.arange(n_data)
            self.rng.shuffle(seq)
            Z[:, :X.shape[1]] = X[seq[:n_inducing_points]]

        self.model.feature.Z = Z

    def train_model(self):

        self.dataset.prepare_data()
        kwargs = self.kwargs
        n_data = self.dataset.data["n_data"]
        n_inducing_points = kwargs["n_inducing_points"]
        n_trajectories = self.dataset.data["n_trajectories"]
        batch_size = kwargs["batch_size"]
        num_batches = max(int((n_trajectories / batch_size)), 1)
        seq = np.arange(n_trajectories)
        if (self.n_iterations == 1) or (n_data <= n_inducing_points):
            self._set_inducing()

        mobj = []
        for step in tqdm(range(kwargs["training_steps"])):

            all_obj = []
            self.rng.shuffle(seq)
            for b in range(int(num_batches)):
                si = b * batch_size
                ei = si + batch_size

                X_b, Y_b, ids_b, ids_unique, psi = self.dataset.get_seq_batch(seq, si, ei)
                data_scale = n_data / X_b.shape[0]
                H_scale = self.n_active_tasks / ids_unique.shape[0]
                feed_dict = {
                    self.model.X_mu_ph: X_b,
                    self.model.Y_ph: Y_b,
                    self.model.data_scale: data_scale,
                    self.model.psi_ph: psi,
                    self.model.H_ids_ph: ids_b,
                    self.model.H_unique_ph: ids_unique,
                    self.model.H_scale: H_scale
                }

                _, obj = self.session.run(
                    [self.model_train_step, self.model_objective],
                    feed_dict=feed_dict)
                all_obj.append(obj)
            if step % self.training_saving_interval == 0:
                mobj.append(np.mean(all_obj))
                logger.info("Step {}/{}: {:.2f}".format(
                    step, kwargs["training_steps"], mobj[-1]))
                self.saver.save(self.session, kwargs["model_path"] + "iter", global_step=step,
                                write_meta_graph=False)

        best_model_index = np.argmin(mobj, axis=0)
        self.saver.restore(self.session, kwargs["model_path"] + "iter-" + str(
            best_model_index * self.training_saving_interval))

    def infer_task_variable(self, env_id, states, actions, test_task_configurations=None):

        kwargs = self.kwargs
        n_data = self.dataset.data["n_data"]

        norm = lambda inp, mu, std: (inp - mu) / std
        states = np.vstack(states)
        actions = np.vstack(actions)
        inputs, outputs = self.dataset.get_inputs_outputs(states, actions)
        if self.kwargs["data_normalization"]:
            inp_norm = norm(inputs, self.dataset.data["inp_mu"], self.dataset.data["inp_std"])
            out_norm = norm(outputs, self.dataset.data["out_mu"], self.dataset.data["out_std"])
        else:
            inp_norm = inputs
            out_norm = outputs
        batch_size = inputs.shape[0]
        ids = np.int32(batch_size * [env_id])
        data_scale = n_data / batch_size
        for step in range(kwargs["latent_variable_inference_steps"]):
            feed_dict = {
                self.model.X_mu_ph: inp_norm,
                self.model.Y_ph: out_norm,
                self.model.data_scale: data_scale,
                self.model.H_ids_ph: ids,
                self.model.H_unique_ph: [env_id],
                self.model.H_scale: self.n_active_tasks,
                self.model.psi_ph: test_task_configurations,
            }

            _, obj = self.session.run(
                [self.model_infer_step, self.model_objective],
                feed_dict=feed_dict)

    def predict_state(self, XH_mu, XH_var, Luu):
        feed_dict = {
            self.trajectory_predictor.XH_mu: XH_mu,
            self.trajectory_predictor.XH_var: XH_var,
            self.trajectory_predictor.Luu: Luu
        }
        mean, var = self.session.run((self.fmean, self.fvar), feed_dict=feed_dict)
        return mean, var

    def get_H_space_subset(self, end_task_id, start_task_id=0):
        H = self.model.H.read_value(session=self.session)
        H_mu = H[start_task_id:end_task_id, :self.dim_h]
        H_var = np.exp(H[start_task_id:end_task_id, self.dim_h:])
        return H_mu, H_var


def init_model(model, objective, session, learning_rate):
    model.initialize(session=session, force=False)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    model_vars = model.trainable_tensors
    model_task_vars = [var for var in model_vars if "H" in var.name]
    train_step = optimizer.minimize(objective, var_list=model_vars)
    task_infer_step = optimizer.minimize(objective, var_list=model_task_vars)

    session.run(tf.variables_initializer(optimizer.variables()))

    return train_step, task_infer_step, optimizer
