import numpy as np


class Fixed1DControlSignalBounds:

    def __init__(self, const_bound: float) -> None:
        self.const_bound = const_bound

    def get_control_signal_bound(self) -> float:
        return self.const_bound

    def get_lower_bound(self):
        return -self.get_control_signal_bound()

    def get_upper_bound(self):
        return self.get_control_signal_bound()


class ControlSignalsGenerator:
    def __init__(self, control_signal_bound_generator,
                 trajectory_length=100) -> None:
        self.trajectory_length = trajectory_length
        self.control_signal_bound_generator = control_signal_bound_generator


class AlternatingControlSignalsGenerator(ControlSignalsGenerator):
    def get_control_signals(self, alternations: int = 10):
        controls = []
        control_signal_bound = self.control_signal_bound_generator.get_control_signal_bound()
        steps_per_chunk = self.trajectory_length / alternations
        for i in range(alternations):
            if i % 2 == 0:
                controls.append(np.arange(control_signal_bound / 2, control_signal_bound,
                                          (control_signal_bound / 2) / steps_per_chunk))
            else:
                controls.append(-1 * np.arange(control_signal_bound / 2, control_signal_bound,
                                               (control_signal_bound / 2) / steps_per_chunk))
        controls = np.array(controls).reshape(self.trajectory_length, 1)
        controls = controls + np.random.normal(scale=1., size=controls.shape)
        return controls


def generate_control_signals(**args):
    control_signal_bounds = Fixed1DControlSignalBounds(const_bound=args["control_signal_upper_bound"])
    controls = None

    if args["policy"] == "ALTERNATE":
        control_signal_generator = AlternatingControlSignalsGenerator(
            control_signal_bound_generator=control_signal_bounds,
            trajectory_length=args["training_trajectory_length"])

        controls = control_signal_generator.get_control_signals(
            alternations=args["alternations"])

    return controls
