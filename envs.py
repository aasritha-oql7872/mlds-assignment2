"""Environment helpers: CartPole-v1 and MsPacman-v5 (with the assignment's preprocessing + frame stacking)."""
from collections import deque
import gymnasium as gym
import numpy as np

mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    """Preprocessing function given in the assignment (crop, downsize, greyscale, contrast, normalize)."""
    img = obs[1:176:2, ::2]                      # crop and downsize
    img = img.sum(axis=2)                        # to greyscale
    img[img == mspacman_color] = 0               # improve contrast
    img = (img // 3 - 128).astype(np.int8)       # normalize from -128 to 127
    return img.reshape(88, 80, 1)


class PacmanEnv(gym.Wrapper):
    """MsPacman-v5 -> stack of the last `n_stack` preprocessed frames, shape (n_stack, 88, 80), int8.

    One frame cannot show which way Pacman / the ghosts are moving, so we stack frames (standard DQN trick).
    """

    def __init__(self, n_stack=4):
        import ale_py
        gym.register_envs(ale_py)
        super().__init__(gym.make("ALE/MsPacman-v5"))        # v5: frameskip=4, sticky actions p=0.25, 9 actions
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.observation_space = gym.spaces.Box(-128, 127, (n_stack, 88, 80), np.int8)

    def _obs(self):
        return np.concatenate(list(self.frames), axis=2).transpose(2, 0, 1)   # (n_stack, 88, 80)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        f = preprocess_observation(obs)
        for _ in range(self.n_stack):                        # edge case: at reset there is no history -> repeat frame 0
            self.frames.append(f)
        return self._obs(), info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        self.frames.append(preprocess_observation(obs))
        return self._obs(), r, terminated, truncated, info


def make_env(name):
    if name == "cartpole":
        return gym.make("CartPole-v1")                       # time limit 500 steps, reward +1 per step
    if name == "pacman":
        return PacmanEnv()
    raise ValueError(name)
