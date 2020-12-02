from env.dm_control_mod.cartpole import swingup, two_poles
from env.dm_control_mod.acrobot import swingup_acrobot
import numpy as np


class EnvironmentConfigurator:

    def __init__(self, env, seed, dt, under_specified_system=False, over_specified_system=False,
                 unobserved_parameter_bounds=None):
        self.env = env
        self.seed = seed
        self.dt = dt
        self.under_specified_system = under_specified_system
        self.over_specified_system = over_specified_system

        self.unobserved_parameter_bounds = unobserved_parameter_bounds

    def get_env(self, configuration):
        env = None
        if self.env == "cartpole":
            if configuration is None:
                env = swingup(random=self.seed, dt=self.dt)
            elif configuration.shape[1] == 1 and not self.under_specified_system:
                env = swingup(random=self.seed, p_m=configuration[0, 0], dt=self.dt)
            elif configuration.shape[1] == 1 and self.under_specified_system:
                sample = np.round(
                    np.random.uniform(self.unobserved_parameter_bounds[0], self.unobserved_parameter_bounds[1]), 2)
                env = swingup(random=self.seed, p_m=sample, p_l=configuration[0, 0], dt=self.dt)
            elif configuration.shape[1] == 2:
                env = swingup(random=self.seed, p_m=configuration[0, 0], p_l=configuration[0, 1], dt=self.dt)
            elif configuration.shape[1] == 3 and self.over_specified_system:
                env = swingup(random=self.seed, p_m=configuration[0, 0], p_l=configuration[0, 1],
                              dt=self.dt)

        elif self.env == "cartdoublepole":
            if configuration is None:
                env = two_poles(random=self.seed, dt=self.dt)
            elif configuration.shape[1] == 2:
                env = two_poles(random=self.seed, p_l_1=configuration[0, 0], p_l_2=configuration[0, 1], dt=self.dt)
            elif configuration.shape[1] == 3:
                env = two_poles(random=self.seed, p_l_1=configuration[0, 0], p_l_2=configuration[0, 1],
                                p_m=configuration[0, 2], dt=self.dt)

        elif self.env == "pendubot":
            if configuration is None:
                env = swingup_acrobot(random=self.seed, dt=self.dt)
            elif configuration.shape[1] == 2:
                env = swingup_acrobot(random=self.seed, p_l_1=configuration[0, 0], p_l_2=configuration[0, 1],
                                      dt=self.dt)
        return env
