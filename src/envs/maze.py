import numpy as np
import d4rl

from spirl.rl.components.environment import GymEnv
from spirl.utils.general_utils import ParamDict


class MazeEnv(GymEnv):
    """Shallow wrapper around gym env for maze envs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._t = 0

    def _default_hparams(self):
        default_dict = ParamDict(
            {
                "start_rand_range": 2.0,  # range of start position randomization, fixed pos if 0.
            }
        )
        return super()._default_hparams().overwrite(default_dict)

    def reset(self):
        super().reset()
        if self.TARGET_POS is not None and self.START_POS is not None:
            start_pos = self.START_POS + self._hp.start_rand_range * (
                np.random.rand(2) * 2 - 1
            )
            self._env.set_target(self.TARGET_POS)
            self._env.reset_to_location(start_pos)
        self._env.render(
            mode="rgb_array"
        )  # these are necessary to make sure new state is rendered on first frame
        self._t = 0
        obs, _, _, _ = self._env.step(np.zeros_like(self._env.action_space.sample()))
        return self._wrap_observation(obs)

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        self._t += 1
        if self._t >= self.max_episode_steps:
            done = True
        if rew > 0:
            rew *= 100.0
            done = True
        return (
            obs,
            np.float64(rew),
            done,
            info,
        )  # casting reward to float64 is important for getting shape later


class ACRandMaze0S40Env(MazeEnv):
    START_POS = np.array([10.0, 24.0])
    TARGET_POS = np.array([18.0, 8.0])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = "maze"
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.max_episode_steps = 2000

    def _default_hparams(self):
        default_dict = ParamDict({"name": "maze2d-randMaze0S40-ac-v0",})
        return super()._default_hparams().overwrite(default_dict)

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        # in the original env 'done' is always set to false
        if self._env.reward_type == "sparse":
            done = done or rew > 0
        else:  # dense reward
            done = done or np.linalg.norm(obs[0:2] - self._env._target) <= 2.0
        return obs, rew, np.array(done), info
