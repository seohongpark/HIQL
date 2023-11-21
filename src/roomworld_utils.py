import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image

def visualize_trajectories(env, trajectories):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    
    env.room.draw()
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for color, trajectory in zip(color_cycle, trajectories):
        all_x = np.array([info['x'] for info in trajectory['info']])
        all_y = np.array([info['y'] for info in trajectory['info']])
        plt.scatter(all_x, all_y, s=5, c=color)
        goal = trajectory['info'][0]['desired_goal']
        plt.scatter(goal[0], goal[1], s=50, c=color, marker='*')
        plt.tight_layout()
    
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def visualize_value(env, value_fn, fig, ax, N=20):
    observations = env.room.XY(n=N)
    values = value_fn(observations)

    x, y = observations[:, 0], observations[:, 1]
    x = x.reshape(N, N)
    y = y.reshape(N, N)
    values = values.reshape(N, N)
    env.room.draw(ax)
    mesh = ax.pcolormesh(x, y, values, cmap='viridis')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(mesh, cax=cax, orientation='vertical')

def visualize_gcvalues(env, value_fn):
    """
    Visualize the value function for a goal-conditioned policy.

    Args:
        env: The environment.
        value_fn: a function with signature value_fn(observations, goal) -> values
    """

    bl, tr = env.room.get_starting_boundary()
    bl, tr = np.array(bl), np.array(tr)

    point1 = bl + (tr - bl) * 0.25
    point2 = bl + (tr - bl) * 0.75
    point3 = np.array([point1[0], point2[1]])
    point4 = np.array([point2[0], point1[1]])

    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    points = [point1, point2, point3, point4]
    for i, point in enumerate(points):
        ax = fig.add_subplot(2, 2, i + 1)
        visualize_value(env, partial(value_fn, goal=point), fig, ax)
        ax.set_title('Goal: ({:.2f}, {:.2f})'.format(point[0], point[1])) 
        ax.scatter(point[0], point[1], s=50, c='red', marker='*')

    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def gc_sampling_adaptor(policy_fn):
    def f(observations, *args, **kwargs):
        return policy_fn(observations['observation'], observations['goal'], *args, **kwargs)
    return f