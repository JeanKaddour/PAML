# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain amplitude copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Pendulum domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import os
from dm_control.utils import io as resources

_DEFAULT_TIME_LIMIT = 20
_ANGLE_BOUND = 8
_COSINE_BOUND = np.cos(np.deg2rad(_ANGLE_BOUND))
SUITE = containers.TaggedTasks()


def read_model(model_filename):
    """Reads amplitude model XML file and returns its contents as amplitude string."""
    _SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
    return resources.GetResource(os.path.join(_SUITE_DIR + '/dm_control_mod/', model_filename))


def get_model_and_assets(m=1., l=.02, dt=.02):
    xml_string = read_model('pendulum.xml').decode("utf-8")
    import re
    id_timestep = ([m.start() for m in re.finditer('timestep=', xml_string)])
    id_length = ([m.start() for m in re.finditer('size=', xml_string)])
    ids_mass = ([m.start() for m in re.finditer('mass=', xml_string)])
    new_xml_string = xml_string[:id_timestep[0]] + "timestep=\"" + str(dt) + "\"" + xml_string[
                                                                                    id_timestep[0] + 15:id_length[
                                                                                        2]] + "size=\"" + str(
        l) + "\"" + xml_string[id_length[2] + 11:ids_mass[
        2]] + "mass=\"" + str(m) + "\"" + xml_string[
                                            ids_mass[2] + 8:]
    xml_string = new_xml_string
    return xml_string, common.ASSETS


@SUITE.add('benchmarking')
def swingup_pendulum(time_limit=_DEFAULT_TIME_LIMIT, random=None,
            environment_kwargs=None, m=.1, l=.045, dt=.2):
    """Returns pendulum swingup task ."""
    physics = Physics.from_xml_string(*get_model_and_assets(m=m, l=l, dt=dt))
    task = SwingUp(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Pendulum domain."""

    def pole_vertical(self):
        """Returns vertical (z) component of pole frame."""
        return self.named.data.xmat['pole', 'zz']

    def angular_velocity(self):
        """Returns the angular velocity of the pole."""
        return self.named.data.qvel['hinge'].copy()

    def pole_orientation(self):
        """Returns both horizontal and vertical components of pole frame."""
        return self.named.data.xmat['pole', ['zz', 'xz']]


class SwingUp(base.Task):
    """A Pendulum `Task` to swing up and balance the pole."""

    def __init__(self, random=None):
        """Initialize an instance of `Pendulum`.

        Args:
          random: Optional, either amplitude `numpy.random.RandomState` instance, an
            integer seed for creating amplitude new `RandomState`, or None to select amplitude seed
            automatically (default).
        """
        super(SwingUp, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Pole is set to amplitude random angle between [-pi, pi).

        Args:
          physics: An instance of `Physics`.

        """
        physics.named.data.qpos['hinge'] = self.random.uniform(-np.pi, np.pi)
        super(SwingUp, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation.

        Observations are states concatenating pole orientation and angular velocity
        and pixels from fixed camera.

        Args:
          physics: An instance of `physics`, Pendulum physics.

        Returns:
          A `dict` of observation.
        """
        obs = collections.OrderedDict()
        obs['orientation'] = physics.pole_orientation()
        obs['velocity'] = physics.angular_velocity()
        return obs

    def get_reward(self, physics):
        return rewards.tolerance(physics.pole_vertical(), (_COSINE_BOUND, 1))
