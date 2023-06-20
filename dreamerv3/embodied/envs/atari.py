import embodied
import numpy as np


"""
The following annotations are obtained from a direct postprocessing of the
annotations proposed in [1], see [2] for the code.

[1]: Anand, A., Racah, E., Ozair, S., Bengio, Y., Côté, M. A., & Hjelm, R. D.
(2019). Unsupervised State Representation Learning in Atari. Advances in Neural
Information Processing Systems, 32.

[2]: <github.com/mila-iqia/atari-representation-learning>: the annotations are
    available at `atariari/benchmark/ram_annotations.py`.
"""

annotations = {
    'asteroids': [73, 74, 60, 61, 62, 83, 84, 86, 87, 89, 90, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37],
    'battle_zone': [46, 47, 48, 52, 53, 54, 58, 105, 84, 4, 59, 60, 108, 29],
    'berzerk': [19, 11, 14, 22, 23, 21, 26, 29, 30, 90, 91, 92, 46, 89, 65, 66, 67, 68, 69, 70, 71, 72, 56, 57, 58, 59, 60, 61, 62, 63, 64, 93, 94, 95],
    'bowling': [30, 41, 29, 40, 36, 33, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
    'boxing': [32, 34, 33, 35, 19, 17, 18],
    'breakout': [99, 101, 72, 77, 84, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    'demonattack': [62, 22, 17, 18, 19, 21, 69, 70, 71, 114],
    'freeway': [14, 103, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117],
    'frostbite': [34, 33, 32, 31, 104, 76, 77, 102, 100, 4, 84, 85, 86, 87, 72, 73, 74],
    'hero': [27, 31, 43, 28, 117, 50, 56, 57],
    'montezuma_revenge': [3, 42, 43, 52, 47, 46, 44, 45, 57, 58, 61, 62, 19, 20, 21],
    'ms_pacman': [6, 7, 8, 9, 12, 13, 14, 15, 10, 16, 11, 17, 19, 56, 119, 120, 123],
    'pitfall': [97, 105, 98, 99, 18, 89, 88],
    'pong': [51, 46, 50, 45, 49, 54, 13, 14],
    'private_eye': [63, 86, 92, 58, 48, 39, 67, 69, 73, 74],
    'qbert': [43, 67, 35, 69, 105, 89, 90, 91, 21, 52, 54, 83, 85, 87, 98, 100, 102, 104, 1, 3, 5, 7, 9, 32, 34, 36, 38, 40, 42],
    'riverraid': [51, 117, 50, 55, 56],
    'seaquest': [70, 97, 86, 87, 102, 103, 59, 62, 30, 31, 32, 33, 71, 72, 73, 74, 57, 58],
    'skiing': [25, 104, 105, 106, 107, 87, 88, 89, 90, 91, 92, 93],
    'spaceinvaders': [17, 104, 73, 28, 26, 9, 24],
    'tennis': [27, 25, 70, 16, 17, 26, 24, 69],
    'venture': [20, 21, 22, 23, 24, 25, 79, 80, 81, 82, 83, 84, 85, 26, 90, 70, 71, 72],
    'videopinball': [67, 68, 98, 102, 48, 50],
    'yarsrevenge': [32, 31, 38, 37, 43, 42, 47, 46],
}


class Atari(embodied.Env):

  LOCK = None

  def __init__(
      self, name, repeat=4, size=(84, 84), gray=True, noops=0, lives='unused',
      sticky=True, actions='all', length=108000, resize='opencv', seed=None,
      flickering=0.0, black=True):
    assert size[0] == size[1]
    assert lives in ('unused', 'discount', 'reset'), lives
    assert actions in ('all', 'needed'), actions
    assert resize in ('opencv', 'pillow'), resize
    assert 0.0 <= flickering <= 1.0
    if self.LOCK is None:
      import multiprocessing as mp
      mp = mp.get_context('spawn')
      self.LOCK = mp.Lock()
    self._resize = resize
    if self._resize == 'opencv':
      import cv2
      self._cv2 = cv2
    if self._resize == 'pillow':
      from PIL import Image
      self._image = Image
    import gym.envs.atari
    if name == 'james_bond':
      name = 'jamesbond'
    self._repeat = repeat
    self._size = size
    self._gray = gray
    self._noops = noops
    self._lives = lives
    self._sticky = sticky
    self._length = length
    self._random = np.random.RandomState(seed)
    self._flickering = flickering
    self._black = black
    self._annotation = annotations[name]
    with self.LOCK:
      self._env = gym.envs.atari.AtariEnv(
          game=name,
          obs_type='image',
          frameskip=1, repeat_action_probability=0.25 if sticky else 0.0,
          full_action_space=(actions == 'all'))
    assert self._env.unwrapped.get_action_meanings()[0] == 'NOOP'
    shape = self._env.observation_space.shape
    self._buffer = [np.zeros(shape, np.uint8) for _ in range(2)]
    self._ale = self._env.unwrapped.ale
    self._last_lives = None
    self._done = True
    self._step = 0

  @property
  def obs_space(self):
    shape = self._size + (1 if self._gray else 3,)
    return {
        'image': embodied.Space(np.uint8, shape),
        'reward': embodied.Space(np.float32),
        'info_ram': embodied.Space(np.float32, (len(self._annotation),)),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      with self.LOCK:
        self._reset()
      self._done = False
      self._step = 0
      return self._obs(0.0, is_first=True)
    total = 0.0
    dead = False
    for repeat in range(self._repeat):
      _, reward, over, info = self._env.step(action['action'])
      self._step += 1
      total += reward
      if repeat == self._repeat - 2:
        self._screen(self._buffer[1])
      if over:
        break
      if self._lives != 'unused':
        current = self._ale.lives()
        if current < self._last_lives:
          dead = True
          self._last_lives = current
          break
    if not self._repeat:
      self._buffer[1][:] = self._buffer[0][:]
    self._screen(self._buffer[0])
    self._done = over or (self._length and self._step >= self._length)
    return self._obs(
        total,
        is_last=self._done or (dead and self._lives == 'reset'),
        is_terminal=dead or over)

  def _reset(self):
    self._env.reset()
    if self._noops:
      for _ in range(self._random.randint(self._noops)):
         _, _, dead, _ = self._env.step(0)
         if dead:
           self._env.reset()
    self._last_lives = self._ale.lives()
    self._screen(self._buffer[0])
    self._buffer[1].fill(0)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    np.maximum(self._buffer[0], self._buffer[1], out=self._buffer[0])
    image = self._buffer[0]
    if image.shape[:2] != self._size:
      if self._resize == 'opencv':
        image = self._cv2.resize(
            image, self._size, interpolation=self._cv2.INTER_AREA)
      if self._resize == 'pillow':
        image = self._image.fromarray(image)
        image = image.resize(self._size, self._image.NEAREST)
        image = np.array(image)
    if self._gray:
      weights = [0.299, 0.587, 1 - (0.299 + 0.587)]
      image = np.tensordot(image, weights, (-1, 0)).astype(image.dtype)
      image = image[:, :, None]

    if self._flickering > 0.0:
        if self._random.random() < self._flickering:
            if self._black:
                image = np.zeros_like(image)
            else:
                image = np.random.rand(*image.shape) * 256

    ale_ram = np.array(self._ale.getRAM()).astype(np.float32) / 128.0
    ale_ram = ale_ram[self._annotation]

    return dict(
        image=image,
        reward=reward,
        info_ram=ale_ram,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_last,
    )

  def _screen(self, array):
    self._ale.getScreenRGB2(array)

  def close(self):
    return self._env.close()
