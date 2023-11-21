import copy
from collections import defaultdict
from contextlib import contextmanager

import hydra
import numpy as np
from gym import spaces
from calvin_env.envs.play_table_env import PlayTableSimEnv


class CalvinEnv(PlayTableSimEnv):
    def __init__(self, tasks: dict = {}, **kwargs):
        self.max_episode_steps = kwargs.pop("max_episode_steps")
        self.reward_norm = kwargs.pop("reward_norm")
        # remove unwanted arguments from the superclass
        [
            kwargs.pop(key)
            for key in [
                "id",
                "screen_size",
                "action_repeat",
                "frame_stack",
                "absorbing_state",
                "pixel_ob",
                "state_ob",
                "num_sequences",
                "data_path",
                "save_dir",
                "record",
            ]
        ]
        super().__init__(**kwargs)

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(21,))

        self.tasks = hydra.utils.instantiate(tasks)
        self.target_tasks = list(self.tasks.tasks.keys())
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        self._t = 0
        self.sequential = False

    def reset(self):
        obs = super().reset()
        self.start_info = self.get_info()
        self._t = 0
        self.tasks_to_complete = copy.deepcopy(self.target_tasks)
        self.completed_tasks = []
        self.solved_subtasks = defaultdict(lambda: 0)
        return obs

    def reset_to_state(self, state):
        return super().reset(robot_obs=state[:15], scene_obs=state[15:])

    def get_obs(self):
        obs = self.get_state_obs()
        return np.concatenate([obs["robot_obs"], obs["scene_obs"]])[:21]

    def _reward(self):
        current_info = self.get_info()
        completed_tasks = self.tasks.get_task_info_for_set(
            self.start_info, current_info, self.target_tasks
        )
        next_task = self.tasks_to_complete[0]

        reward = 0
        for task in list(completed_tasks):
            if self.sequential:
                if task == next_task:
                    reward += 1
                    self.tasks_to_complete.pop(0)
                    self.completed_tasks.append(task)
            else:
                if task in self.tasks_to_complete:
                    reward += 1
                    self.tasks_to_complete.remove(task)
                    self.completed_tasks.append(task)

        reward *= self.reward_norm
        r_info = {"reward": reward}
        return reward, r_info

    def _termination(self):
        """Indicates if the robot has completed all tasks. Should be called after _reward()."""
        done = len(self.tasks_to_complete) == 0
        d_info = {"success": done}
        return done, d_info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        for task in self.target_tasks:
            self.solved_subtasks[task] = (
                1 if task in self.completed_tasks or self.solved_subtasks[task] else 0
            )
        return info

    def step(self, action):
        """Performing a relative action in the environment
        input:
            action: 7 tuple containing
                    Position x, y, z.
                    Angle in rad x, y, z.
                    Gripper action
                    each value in range (-1, 1)
        output:
            observation, reward, done info
        """
        # Transform gripper action to discrete space
        env_action = action.copy()
        env_action[-1] = (int(action[-1] >= 0) * 2) - 1
        self.robot.apply_action(env_action)
        for _ in range(self.action_repeat):
            self.p.stepSimulation(physicsClientId=self.cid)
        self.scene.step()
        obs = self.get_obs()
        info = self.get_info()
        reward, r_info = self._reward()
        done, d_info = self._termination()
        info.update(r_info)
        info.update(d_info)
        self._t += 1
        if self._t >= self.max_episode_steps:
            done = True
        return obs, reward, done, self._postprocess_info(info)

    @contextmanager
    def val_mode(self):
        """Sets validation parameters if desired. To be used like: with env.val_mode(): ...<do something>..."""
        pass
        yield
        pass

    def get_episode_info(self):
        completed_tasks = (
            self.completed_tasks if len(self.completed_tasks) > 0 else [None]
        )
        info = dict(
            solved_subtask=completed_tasks, tasks_to_complete=self.tasks_to_complete
        )
        info.update(self.solved_subtasks)
        return info
