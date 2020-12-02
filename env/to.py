from env.environment_configurator import EnvironmentConfigurator
import numpy as np
from tqdm import tqdm


class TrajectoryObserver:
    def __init__(self, env_name, seed, dt, under_specified_system=False, over_specified_system=False,
                 unobserved_parameter_bounds=None):
        self.env = env_name
        self.env_wrapper = EnvironmentConfigurator(env_name, seed, dt, under_specified_system, over_specified_system,
                                                   unobserved_parameter_bounds)

    def observe_trajectories(self, task_configurations, controls, dim_states=4):
        observations = np.zeros((task_configurations.shape[0], controls.shape[0], dim_states))
        for i in tqdm(range(task_configurations.shape[0])):
            env = self.env_wrapper.get_env(task_configurations[i][None, :])
            observations[i] = self.observe(env, controls)
        return observations

    def observe(self, env, controls):
        observations = None
        trajectory_length = len(controls)
        if self.env == "cartpole":
            time_step = env.reset()
            positions = time_step[3]['position']
            velocities = time_step[3]['velocity']
            observations = np.zeros((trajectory_length, 4))
            theta = env.physics.named.data.qpos[1:]
            cart_pos = positions[0]
            cart_vel = velocities[0]
            theta_val = velocities[1]
            observations[0] = np.hstack([theta, cart_pos, cart_vel, theta_val])
            for i in range(1, trajectory_length):
                action = controls[i - 1]
                time_step = env.step(action)
                positions = time_step[3]['position']
                velocities = time_step[3]['velocity']
                theta = env.physics.named.data.qpos[1]
                cart_pos = positions[0]
                cart_vel = velocities[0]
                theta_val = velocities[1]
                observations[i] = np.hstack([theta, cart_pos, cart_vel, theta_val])
        elif self.env == "cartdoublepole":
            observations = np.zeros((trajectory_length, 6))
            theta_1 = np.clip(env.physics.named.data.qpos[1], -2 * np.pi, 2 * np.pi)
            theta_2 = np.arccos(env.physics.named.data.xmat[3, 0])
            state = np.hstack([theta_1,
                               theta_2,
                               env.physics.named.data.qpos[0],
                               env.physics.named.data.qvel])
            observations[0] = state
            for i in range(1, trajectory_length):
                action = controls[i - 1]
                _ = env.step(action)
                observations[i] = np.hstack([env.physics.named.data.qpos[1],
                                             np.arccos(env.physics.named.data.xmat[3, 0]),
                                             env.physics.named.data.qpos[0],
                                             env.physics.named.data.qvel])
        elif self.env == "pendubot":
            time_step = env.reset()
            observations = np.zeros((trajectory_length, 4))
            theta_1 = np.arcsin(env.physics.named.data.xmat[['upper_arm', 'lower_arm'], 'xz'][0])
            theta_2 = np.arcsin(env.physics.named.data.xmat[['upper_arm', 'lower_arm'], 'xz'][1])
            velocities = time_step[3]['velocity']
            state = np.hstack([theta_1,
                               theta_2,
                               velocities])
            observations[0] = state
            for i in range(1, trajectory_length):
                action = controls[i - 1]
                time_step = env.step(action)
                theta_1 = np.arcsin(env.physics.named.data.xmat[['upper_arm', 'lower_arm'], 'xz'][0])
                theta_2 = np.arcsin(env.physics.named.data.xmat[['upper_arm', 'lower_arm'], 'xz'][1])
                velocities = time_step[3]['velocity']
                observations[i] = np.hstack([theta_1,
                                             theta_2,
                                             velocities])
        return observations

    def get_start_state(self):
        start_state = None
        env = self.env_wrapper.get_env(None)
        if self.env == "cartpole":
            time_step = env.reset()
            positions = time_step[3]['position']
            velocities = time_step[3]['velocity']
            theta = env.physics.named.data.qpos[1:]
            cart_pos = positions[0]
            cart_vel = velocities[0]
            theta_val = velocities[1]
            start_state = np.hstack([theta, cart_pos, cart_vel, theta_val]).reshape(1, -1)
        elif self.env == "cartdoublepole":
            env.reset()
            start_state = np.hstack([env.physics.named.data.qpos[1],
                                     np.arccos(env.physics.named.data.xmat[3, 0]),
                                     env.physics.named.data.qpos[0],
                                     env.physics.named.data.qvel]).reshape(1, -1)
        elif self.env == "pendubot":
            time_step = env.reset()
            velocities = time_step[3]['velocity']
            start_state = np.hstack([0.0,
                                     0.0,
                                     velocities]).reshape(1, -1)
        return start_state
