from utils.math_utils import sample_from_multidim_interval_uniformly


def acquire_random_task_descriptor(configuration_space_interval):
    return sample_from_multidim_interval_uniformly(bounds=configuration_space_interval, size=1)
