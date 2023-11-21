from collections import defaultdict

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
from scipy.special import softmax
import matplotlib
import matplotlib.pyplot as plt

import functools as ft
from procgen import ProcgenEnv
import jax
import jax.numpy as jnp


def get_xy(obs):
    grey = np.array([187, 203, 204])
    dists = np.abs(obs - grey).sum(axis=-1)
    return np.stack([
        dists.min(-2).argmin(-1), dists.min(-1).argmin(-1)
    ], axis=1)


def get_xy_single(ob):
    return get_xy(np.expand_dims(ob, 0))[0]


def imshow(x, **kwargs):
    plt.imshow(x, **kwargs)


def create_env_for_level(level):
    return ProcgenEnv(num_envs=1,
                              env_name='maze',
                              distribution_mode='easy',
                              start_level=level,
                              num_levels=1)


def get_all_states(level):
    env = create_env_for_level(level)
    init_images = env.reset()
    init_images = init_images['rgb']
    position = tuple(get_xy(init_images)[0])
    start_state = env.env.callmethod('get_state')
    to_process_states = {position: start_state}
    seen_states = {position: (start_state, init_images[0])}
    edges = defaultdict(set)

    while len(to_process_states) > 0:
        position, state = next(iter(to_process_states.items()))
        for action in [2, 3, 5, 6]:
            anti_action = {2: 6, 3: 5, 5: 3, 6: 2}[action]
            env.env.callmethod('set_state', state)
            images, _, done, _ = env.step(np.full((1,), action))
            images = images['rgb']
            if done[0]:
                continue
            new_position = tuple(get_xy(images)[0])
            if new_position in seen_states:
                continue
            if any([abs(new_position[0] - x) + abs(new_position[1] - y) < 3 for x, y in seen_states]):
                continue
            else:
                new_state = env.env.callmethod('get_state')
                to_process_states[new_position] = new_state
                seen_states[new_position] = (new_state, images[0])
                edges[position].add((action, new_position))
                edges[new_position].add((anti_action, position))
        del to_process_states[position]
    return seen_states, edges


def probs_to_values(probs):
    # 0, 1, 2: Left
    # 3, Down
    # 5 Up
    # 6, 7, 8 Right
    # 4, 9, 10, 11, 12, 13, 14: Nothing
    left = probs[..., [0, 1, 2]].sum(-1)
    down = probs[..., 3]
    up = probs[..., 5]
    right = probs[..., [6, 7, 8]].sum(-1)
    none = probs[..., [4, 9, 10, 11, 12, 13, 14]].sum(-1)
    return np.stack([none, left, down, right, up], axis=-1)


def get_polygon(x, y, action):
    scale = 4
    if action == 0:
        return np.array([
            [x - 0.2 * scale, y + 0.2 * scale],
            [x + 0.2 * scale, y + 0.2 * scale],
            [x + 0.2 * scale, y - 0.2 * scale],
            [x - 0.2 * scale, y - 0.2 * scale]
        ])
    center = np.array([x, y])
    tl = np.array([x - 0.5 * scale, y + 0.5 * scale])
    tr = np.array([x + 0.5 * scale, y + 0.5 * scale])
    bl = np.array([x - 0.5 * scale, y - 0.5 * scale])
    br = np.array([x + 0.5 * scale, y - 0.5 * scale])
    if action == 1:  # Left
        return np.stack([center, bl, tl])
    if action == 2:  # Down
        return np.stack([center, tl, tr])
    if action == 3:  # right
        return np.stack([center, tr, br])
    if action == 4:  # Up
        return np.stack([center, br, bl])


def draw_q(positions, q, ax=None, cm=plt.cm.Blues):
    if ax is None: ax = plt.gca()
    norm = matplotlib.colors.Normalize(vmin=q.min(), vmax=q[:, 1:].max())
    for idx in range(len(positions)):
        x, y = positions[idx]
        for action in [1, 2, 3, 4, 0]:
            t1 = plt.Polygon(get_polygon(x, y, action), color=cm(norm(q[idx, action])), lw=0.1, ec='black', alpha=1.0)
            ax.add_patch(t1)
    ax.set_xlim(0, 64)
    ax.set_ylim(64, 0)
    return norm


from dataclasses import dataclass


@dataclass
class ProcgenLevel:
    states: dict
    locs: np.array
    imgs: np.array
    tuple_locs: list
    edges: dict
    distances: dict
    optimal_actions: dict

    @classmethod
    def create(cls, level_id):
        all_states, all_edges = get_all_states(level_id)
        tuple_locs = list(all_states.keys())
        all_locs = np.array(list(all_states.keys()))
        all_imgs = np.array([all_states[tuple(loc)][1] for loc in all_locs])

        distances = defaultdict(dict)
        optimal_actions = defaultdict(dict)
        action_to_idx = {4: 0, 2: 1, 3: 2, 6: 3, 5: 4}
        for start_state in all_edges.keys():
            queue = []
            distances[start_state][start_state] = 0
            optimal_actions[start_state][start_state] = action_to_idx[4]
            queue.append(start_state)
            cur = 0
            while cur < len(queue):
                cur_state = queue[cur]
                for action, next_state in all_edges[cur_state]:
                    if next_state in distances[start_state]:
                        continue
                    distances[start_state][next_state] = distances[start_state][cur_state] + 1
                    optimal_actions[start_state][next_state] = action_to_idx[action] if optimal_actions[start_state][cur_state] == 0 else optimal_actions[start_state][cur_state]
                    queue.append(next_state)
                cur += 1

        return cls(all_states, all_locs, all_imgs, tuple_locs, all_edges, distances, optimal_actions)


def plot_value_fn(level, value_fn, fig, ax, cmap=plt.cm.bwr):
    values = value_fn(level)
    ax.imshow(level.imgs[0], alpha=0.25)
    data = ax.scatter(*level.locs.T, c=values, cmap=cmap)
    fig.colorbar(data, ax=ax)


def plot_policy_fn(level, policy_fn, ax, cmap=plt.cm.Blues):
    ax.imshow(level.imgs[0])
    probs = policy_fn(level)
    draw_q(level.locs, probs, ax=ax, cm=cmap)


def get_canvas_image(canvas):
    canvas.draw()
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image


def make_visualization(level_id, q_fn_cur, q_fn_old, dataset, level_metrics):
    level = ProcgenLevel.create(level_id)

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    canvas = FigureCanvas(fig)

    # Plot dataset
    axes[0][0].set_title('Dataset')
    axes[0][0].imshow(level.imgs[0], alpha=0.25)
    xy = get_xy(dataset['observations'])
    axes[0][0].scatter(*xy.T, alpha=0.1)

    def value(q_fn, level):
        return q_fn(level.imgs).max(-1)

    def policy(q_fn, level):
        q = q_fn(level.imgs)
        policy = np.array(softmax(q, axis=-1))
        return probs_to_values(policy)

    # Plot value
    axes[0][1].set_title('Value')
    plot_value_fn(level, ft.partial(value, q_fn_cur), fig, axes[0][1])

    # Plot policy
    axes[0][2].set_title('Policy')
    plot_policy_fn(level, ft.partial(policy, q_fn_cur), axes[0][2])

    # Plot value diff
    axes[1][0].set_title('Value Diff')
    plot_value_fn(level, lambda level: value(q_fn_cur, level) - value(q_fn_old, level), fig, axes[1][0],
                  cmap=plt.cm.bwr)

    # Plot policy diff
    axes[1][1].set_title('Policy Diff')
    plot_policy_fn(level, lambda level: policy(q_fn_cur, level) - policy(q_fn_old, level), axes[1][1], cmap=plt.cm.bwr)

    def plot_metric(metric_key, ax, title):
        ax.plot(level_metrics[metric_key])
        ax.set_title(title)

    plot_metric('online/critic_loss', axes[2][0], 'Online Critic Loss')
    plot_metric('online/q_gap', axes[2][1], 'Online Q Gap')
    plot_metric('evaluation/mean return', axes[2][2], 'Return')
    plot_metric('offline/critic_loss', axes[3][0], 'Offline Critic Loss')
    plot_metric('offline/q_gap', axes[3][1], 'Offline Q Gap')

    plt.tight_layout()
    image = get_canvas_image(canvas)
    fig.clf()
    plt.close(fig)
    return image
