import functools
import os

import embodied
import numpy as np


class DMC(embodied.Env):

  DEFAULT_CAMERAS = dict(
      locom_rodent=1,
      quadruped=2,
  )

  def __init__(self, env, repeat=1, render=True, size=(64, 64), camera=-1, flickering=0.0, seed=None, black=True):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    assert 0.0 <= flickering <= 1.0
    if 'MUJOCO_GL' not in os.environ:
      os.environ['MUJOCO_GL'] = 'egl'
    if isinstance(env, str):
      domain, task = env.split('_', 1)
      if camera == -1:
        camera = self.DEFAULT_CAMERAS.get(domain, 0)
      if domain == 'cup':  # Only domain with multiple words.
        domain = 'ball_in_cup'
      if domain == 'manip':
        from dm_control import manipulation
        env = manipulation.load(task + '_vision')
      elif domain == 'locom':
        from dm_control.locomotion.examples import basic_rodent_2020
        env = getattr(basic_rodent_2020, task)()
      else:
        from dm_control import suite
        env = suite.load(domain, task)
    self._dmenv = env
    from . import from_dm
    self._env = from_dm.FromDM(self._dmenv)
    self._env = embodied.wrappers.ExpandScalars(self._env)
    self._env = embodied.wrappers.ActionRepeat(self._env, repeat)
    self._render = render
    self._size = size
    self._camera = camera
    self._flickering = flickering
    self._black = black
    self._random = np.random.RandomState(seed)

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    infos = {}
    for k, v in spaces.items():
      if k not in ('reward', 'is_first', 'is_last', 'is_terminal'):
        infos['info_' + k] = v
    spaces |= infos
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])
    obs = self._env.step(action)
    infos = {}
    for k, v in obs.items():
        if k not in ('reward', 'is_first', 'is_last', 'is_terminal'):
            infos['info_' + k] = v
    obs |= infos
    if self._render:
      obs['image'] = self.render()
      if self._flickering > 0.0:
          if self._random.random() < self._flickering:
              if self._black:
                  obs['image'] = np.zeros_like(obs['image'])
              else:
                  obs['image'] = np.random.rand(*obs['image'].shape) * 256

    return obs

  def render(self):
    return self._dmenv.physics.render(*self._size, camera_id=self._camera)
