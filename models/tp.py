import tensorflow as tf
from gpflow import settings
from gpflow.params import Parameterized


class TrajectoryPredictor(Parameterized):

    def __init__(self, model, dim_states, dim_actions, dim_angles, trajectory_length,
                 dim_h, inducing_points):
        super(TrajectoryPredictor, self).__init__()

        self.model = model
        self.dim_angles = dim_angles
        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.dim_h = dim_h
        self.trajectory_length = trajectory_length
        self.inducing_points = inducing_points

        self.XH_mu = None
        self.XH_var = None

        self.init_XH()

        self.H_mu = tf.placeholder(
            settings.float_type, [1, self.dim_h])
        self.H_var = tf.placeholder(
            settings.float_type, [1, self.dim_h, self.dim_h])
        self.Luu = tf.placeholder(
            settings.float_type, [self.inducing_points, self.inducing_points])

    def init_XH(self):
        self.XH_mu = tf.placeholder(
            settings.float_type,
            [self.trajectory_length - 1, self.dim_states + self.dim_angles + self.dim_actions + self.dim_h])
        self.XH_var = tf.placeholder(
            settings.float_type,
            [self.trajectory_length - 1, self.dim_states + self.dim_angles + self.dim_actions + self.dim_h,
             self.dim_states + self.dim_angles + self.dim_actions + self.dim_h])

    def predict_state(self):
        mean, var, _ = self.model.build_predict_uncertain(self.XH_mu,
                                                          self.XH_var,
                                                          full_cov=False, Luu=self.Luu)
        return mean, var
