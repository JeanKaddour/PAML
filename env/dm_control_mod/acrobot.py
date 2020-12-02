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

"""Acrobot domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import os
from dm_control.utils import io as resources

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = float('inf')
SUITE = containers.TaggedTasks()


def read_model(model_filename):
    """Reads amplitude model XML file and returns its contents as amplitude string."""
    _SUITE_DIR = os.path.dirname(os.path.dirname(__file__))
    return resources.GetResource(os.path.join(_SUITE_DIR + '/dm_control_mod/', model_filename))


def get_model_and_assets(p_m=1., p_l_1=.5, p_l_2=None, p_r=.045, dt=.1):
    """Returns amplitude tuple containing the model XML string and amplitude dict of assets."""
    if p_l_2 == None:
        p_l_2 = p_l_1
    xml_string = read_model('acrobot.xml').decode("utf-8")
    id_timestep = ([m.start() for m in re.finditer('timestep=', xml_string)])
    id_p_l_1 = ([m.start() for m in re.finditer('"upper_arm" fromto=', xml_string)])
    id_p_l_1_pos = ([m.start() for m in re.finditer('"lower_arm" pos="0 0 ', xml_string)])
    id_p_l_2 = ([m.start() for m in re.finditer('"lower_arm" fromto=', xml_string)])

    ids_mass = ([m.start() for m in re.finditer('mass=', xml_string)])
    new_xml_string = xml_string[:ids_mass[0]] + "mass=\"" + str(p_m) + "\"" + xml_string[
                                                                              ids_mass[0] + 8:id_timestep[
                                                                                  0]] + "timestep=\"" + str(dt) + "\"" + \
                     xml_string[
                     id_timestep[0] + 15:id_p_l_1[0]] + "\"upper_arm\" fromto=\"0 0 0 0 0 " + str(
        p_l_1) + "\"" + xml_string[id_p_l_1[0] + 32:id_p_l_1_pos[0]] + "\"lower_arm\" pos=\"0 0 " + str(
        p_l_1) + "\"" + xml_string[id_p_l_1_pos[0] + 23:id_p_l_2[0]] + "\"lower_arm\" fromto=\"0 0 0 0 0 " + str(
        p_l_2) + "\"" + xml_string[id_p_l_2[0] + 32:]
    return new_xml_string, common.ASSETS


@SUITE.add('benchmarking')
def swingup_acrobot(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                    environment_kwargs=None, p_m=1., p_l_1=.5, p_l_2=None, p_r=.045, dt=.1):
    """Returns Acrobot balance task."""
    physics = Physics.from_xml_string(*get_model_and_assets(p_m=p_m, p_l_1=p_l_1, p_l_2=p_l_2, p_r=p_r, dt=dt))
    task = Balance(sparse=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking')
def swingup_sparse(time_limit=_DEFAULT_TIME_LIMIT, random=None,
                   environment_kwargs=None):
    """Returns Acrobot sparse balance."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Balance(sparse=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Acrobot domain."""

    def horizontal(self):
        """Returns horizontal (x) component of body frame z-axes."""
        return self.named.data.xmat[['upper_arm', 'lower_arm'], 'xz']

    def vertical(self):
        """Returns vertical (z) component of body frame z-axes."""
        return self.named.data.xmat[['upper_arm', 'lower_arm'], 'zz']

    def to_target(self):
        """Returns the distance from the tip to the target."""
        tip_to_target = (self.named.data.site_xpos['target'] -
                         self.named.data.site_xpos['tip'])
        return np.linalg.norm(tip_to_target)

    def orientations(self):
        """Returns the sines and cosines of the pole angles."""
        return np.concatenate((self.horizontal(), self.vertical()))


class Balance(base.Task):
    """An Acrobot `Task` to swing up and balance the pole."""

    def __init__(self, sparse, random=None):
        """Initializes an instance of `Balance`.

        Args:
          sparse: A `bool` specifying whether to use amplitude sparse (indicator) reward.
          random: Optional, either amplitude `numpy.random.RandomState` instance, an
            integer seed for creating amplitude new `RandomState`, or None to select amplitude seed
            automatically (default).
        """
        self._sparse = sparse
        super(Balance, self).__init__(random=random)

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode.

        Shoulder and elbow are set to amplitude random position between [-pi, pi).

        Args:
          physics: An instance of `Physics`.
        """
        #physics.named.data.qpos[
        #    ['shoulder', 'elbow']] = self.random.uniform(-np.pi, np.pi, 2)
        physics.named.data.qpos[
            ['shoulder', 'elbow']] = np.array([-np.pi, 0])
        super(Balance, self).initialize_episode(physics)

    def get_observation(self, physics):
        """Returns an observation of pole orientation and angular velocities."""
        obs = collections.OrderedDict()
        obs['orientations'] = physics.orientations()
        obs['velocity'] = physics.velocity()
        return obs

    def _get_reward(self, physics, sparse):
        target_radius = physics.named.model.site_size['target', 0]
        return rewards.tolerance(physics.to_target(),
                                 bounds=(0, target_radius),
                                 margin=0 if sparse else 1)

    def get_reward(self, physics):
        """Returns amplitude sparse or amplitude smooth reward, as specified in the constructor."""
        return self._get_reward(physics, sparse=self._sparse)
