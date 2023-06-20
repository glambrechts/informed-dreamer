import random
import embodied
import numpy as np

from math import pi, cos, sin
from scipy.stats import multivariate_normal as mvn


# Special states
START_POSITION = np.array([-0.8, -0.8], dtype=np.float32)
GOAL_POSITION = np.array([0.8, 0.8], dtype=np.float32)
LOWER_BOUND = np.array([-1.0, -1.0], dtype=np.float32)
UPPER_BOUND = np.array([1.0, 1.0], dtype=np.float32)


# Altitude function components
MVN_1 = mvn(mean=np.array([0.0, 0.0], dtype=np.float32), cov=np.array([[0.005, 0.0], [0.0, 1.0]], dtype=np.float32))
MVN_2 = mvn(mean=np.array([0.0, -0.8], dtype=np.float32), cov=np.array([[1.0, 0.0], [0.0, 0.01]], dtype=np.float32))
MVN_3 = mvn(mean=np.array([0.0, 0.8], dtype=np.float32), cov=np.array([[1.0, 0.0], [0.0, 0.01]], dtype=np.float32))
SLOPE = np.array([0.2, 0.2], dtype=np.float32)


# Rotation matrix
ROTATION_90 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)


class MountainHike(embodied.Env):

  def __init__(self, task, step_size=0.1, transition_std=0.05, observation_std=0.1, discrete=False, altitude=False, rotations=False, length=200):

    self.step_size = step_size
    self.transition_std = transition_std
    self.observation_std = observation_std

    self.discrete = discrete
    self.altitude = altitude
    self.rotations = rotations

    self.low = - step_size
    self.high = step_size

    self._task = task
    self._length = length

    # Reset
    self._init()

  def _init(self):
    self._position = np.copy(START_POSITION)

    if self.rotations:
      if self.discrete:
        self._rotation = random.randrange(4)
        self._rotation_matrix = np.linalg.matrix_power(ROTATION_90, self._rotation)
      else:
        self._rotation = random.randrange(4)
        self._rotation_matrix = np.linalg.matrix_power(ROTATION_90, self._rotation)
    else:
      self._rotation = None
      self._rotation_matrix = np.eye(2)

    if self.discrete:
      self._moves = [
        np.array([self.step_size, 0.0]),
        np.array([0.0, self.step_size]),
        np.array([- self.step_size, 0.0]),
        np.array([0.0, - self.step_size])
      ]

    self._done = False
    self._step = 0

  @property
  def obs_space(self):

    return {
        'vector': embodied.Space(np.float32, (1 if self.altitude else 2,)),
        'info_position': embodied.Space(np.float32, (3 if self.rotations else 2,)),
        'step': embodied.Space(np.int32, (), 0, self._length),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    if self.discrete:
      space = embodied.Space(np.int32, (), 0, 4)
    else:
      space = embodied.Space(np.float32, (2,), - self.step_size, self.step_size)
    return {'action': space, 'reset': embodied.Space(bool)}

  def _altitude(self, position=None):
    """
    Returns the altitude at the current position (x, y).
    """
    if position is None:
      position = self._position.reshape(1, -1)

    mountains = [MVN_1.pdf(position), MVN_2.pdf(position), MVN_3.pdf(position)]

    altitude = np.maximum.reduce(mountains)
    if np.isscalar(altitude):
      altitude = np.array(altitude)

    return (-np.exp(-altitude)) + (position @ SLOPE) - 0.02

  def _observation(self):
    if self.altitude:
      return self._altitude() + np.random.randn(1).astype(np.float32) * self.observation_std
    else:
      return self._position + np.random.randn(2).astype(np.float32) * self.observation_std

  def _reward(self):
    if self._terminal():
      return 0.0
    return float(self._altitude())

  def _terminal(self, last=False):
    position = self._last_position if last else self._position
    return np.linalg.norm(position - GOAL_POSITION) < self.step_size * 2

  def _transition(self, action):
    self._last_position = np.copy(self._position)

    if self._terminal():
      return

    if self.discrete:
      self._position += self._rotation_matrix @ self._moves[action]
    else:
      self._position += self._rotation_matrix @ action

    self._position += np.random.randn(2).astype(np.float32) * self.transition_std
    self._position = np.clip(self._position, LOWER_BOUND, UPPER_BOUND)

  def step(self, action):

    if action['reset'] or self._done:
      self._init()
      observation = self._observation()
      return self._obs(observation, 0.0, is_first=True)
    action = action['action']

    if not self.discrete:
      action = np.clip(action, self.low, self.high)

    self._transition(action)
    reward = self._reward()
    observation = self._observation()
    terminal = self._terminal()

    self._step += 1
    self._done = (self._step >= self._length or terminal)


    return self._obs(observation, reward, is_last=self._done, is_terminal=self._done)

  def state(self):
    if self.rotations:
      return np.array([self._position[0], self._position[1], self._rotation], dtype=np.float32)
    else:
      return self._position.copy()


  def _obs(self, observation, reward, is_first=False, is_last=False, is_terminal=False):

    return dict(
        vector=observation,
        info_position=self.state(),
        step=self._step,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )
