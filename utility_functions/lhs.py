from pyDOE2 import lhs


def latin_hypercube_sampling_maxi_min(n_tasks: int, dim: int, **args):
    task_configurations = lhs(dim, n_tasks, criterion="centermaximin", random_state=args["seed"])
    lower_bound = args["observed_configuration_space_interval"][:, 0]
    upper_bound = args["observed_configuration_space_interval"][:, 1]
    task_configurations = (upper_bound - lower_bound) * task_configurations + lower_bound
    return task_configurations
