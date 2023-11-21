import numpy as np
from procgen import ProcgenEnv

from jaxrl_m.dataset import Dataset


class ProcgenWrappedEnv:
    def __init__(self, num_envs, env_name, start_level, num_levels, distribution_mode='easy'):
        self.envs = ProcgenEnv(num_envs=num_envs,
                               env_name=env_name,
                               distribution_mode=distribution_mode,
                               start_level=start_level,
                               num_levels=num_levels)
        self.r = np.zeros(num_envs)
        self.t = np.zeros(num_envs)

    def obs(self, x):
        return x['rgb']

    def reset(self):
        self.r = np.zeros_like(self.r)
        self.t = np.zeros_like(self.r)
        return self.obs(self.envs.reset())

    def step(self, a):
        obs, r, done, info = self.envs.step(a)
        self.r += r
        self.t += 1
        for n in range(len(done)):
            if done[n]:
                info[n]['episode'] = dict(r=self.r[n], t=self.t[n], time_r=max(-500, -1 * self.t[n]))
                self.r[n] = 0
                self.t[n] = 0
        return self.obs(obs), r, done, info


def get_procgen_dataset(buffer_fname, state_based=False):

    print(f'Attempting to load buffer: {buffer_fname}')
    buffer = np.load(buffer_fname)
    buffer = {k: buffer[k] for k in buffer}
    print(f'Loaded buffer with {len(buffer["action"])} observations')

    env_infos = dict(level_id=buffer['prev_level_seed'])
    print(buffer.keys())
    if 'qstar' in buffer:
        env_infos = dict(
            level_id=buffer['prev_level_seed'],
            qstar=buffer['qstar'],
        )
        print('Using QSTAR')

    if buffer['observation'].shape[-1] != 3:
        print('Assuming observation is in CHW format')
        buffer['observation'] = np.moveaxis(buffer['observation'], 1, -1)
        buffer['next_observation'] = np.moveaxis(buffer['next_observation'], 1, -1)

    # next_observations are incorrect -> drop last transitions
    buffer['done'][-1] = 1
    non_last_idx = np.nonzero(~buffer['done'])[0]
    last_idx = np.nonzero(buffer['done'])[0]
    penult_idx = last_idx - 1

    good_idx = non_last_idx

    new_dataset = dict()
    for k, v in buffer.items():
        if k == 'done':
            v[penult_idx] = 1
        new_dataset[k] = v[good_idx]
    env_infos = {k: v[good_idx] for k, v in env_infos.items()}
    buffer = new_dataset

    return Dataset.create(
        observations=buffer['observation'] if not state_based else buffer['position'],
        actions=buffer['action'],
        rewards=buffer['reward'],
        masks=np.logical_not(buffer['done']),
        next_observations=buffer['next_observation'] if not state_based else buffer['next_position'],
        # env_infos=env_infos
    )


def bootstrap_std(arr, f=np.mean, n=30):
    arr = np.array(arr)
    return np.std([
        f(arr[np.random.choice(len(arr), len(arr))])
        for _ in range(n)
    ])