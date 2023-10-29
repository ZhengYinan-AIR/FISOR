import os
import sys
sys.path.append('.')
from absl import app, flags
import re
import numpy as np
from ml_collections import config_flags
import matplotlib.pyplot as plt
from matplotlib import colors
import jax
from env.point_robot import PointRobot
from jaxrl5.agents import FISOR


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_integer('seed', -1, '')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
config_flags.DEFINE_config_file(
    "config",
    "configs/train_config.py:fisor",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

hazard_position_list = [np.array([0.4, -1.2]), np.array([-0.4, 1.2])]

label_size = 18
legend_size = 30
ticks_size = 18
location = -0.3
width = 0.5

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': label_size}

def plot_pr_pic(ax, agent, v, theta, cb=False):
    ## generate batch obses
    x1 = np.linspace(-3.0, 3.0, 201)
    x2 = np.linspace(-3.0, 3.0, 201)
    x1_grid, x2_grid = np.meshgrid(x1, x2)
    flatten_x1 = x1_grid.ravel()
    flatten_x2 = x2_grid.ravel()
    batch_obses = np.zeros((len(flatten_x1), 11), dtype=np.float32)  # (201*201, 11)
    assert batch_obses.shape == (201*201, 11)
    batch_obses[:, 0] = flatten_x1
    batch_obses[:, 1] = flatten_x2

    batch_obses[:, 2] = v * np.ones_like(flatten_x1)
    thetas = theta * np.ones_like(flatten_x1)
    batch_obses[:, 3] = np.cos(thetas)
    batch_obses[:, 4] = np.sin(thetas)

    c = np.cos(theta)
    s = np.sin(theta)

    rot_mat = np.array([[c, -s],
                        [s, c]], dtype=np.float32)

    k = 0

    for hazard_pos in hazard_position_list:

        pos = (hazard_pos[:2] - batch_obses[:,:2]) @ rot_mat  # (B, 2)
        x = pos[:,0]
        y = pos[:,1]
        hazard_vec = x + 1j * y

        dist = np.abs(hazard_vec)
        angle = np.angle(hazard_vec)

        batch_obses[:,5+k*3] = dist
        batch_obses[:,6+k*3] = np.cos(angle)
        batch_obses[:,7+k*3] = np.sin(angle)

        k += 1
    
    '''
    safe value
    '''
    safe_value = agent.safe_value.apply_fn({"params": agent.safe_value.params}, jax.device_put(batch_obses))
    value = safe_value

    value_flatten = np.asarray(value)
    value_square = value_flatten.reshape(x1_grid.shape)
        
    '''
    draw hj
    '''
    norm = colors.Normalize(vmin=-3.5, vmax=1.01)
    
    ct = ax.contourf(
        x1_grid, x2_grid, value_square,
        norm=norm,
        levels=30,
        cmap='rainbow',
    )

    ct_line = ax.contour(
        x1_grid, x2_grid, value_square,
        levels=[0], colors='#32ABD6',
        linewidths=2.0, linestyles='solid'
    )
    ax.clabel(ct_line, inline=True, fontsize=15, fmt=r'0',)

    if cb==True:
        cb = plt.colorbar(ct, ax=ax, shrink=0.8, pad=0.02, ticks=np.linspace(-3.2, 0.8, 6))
        cb.ax.tick_params(labelsize=ticks_size)

        cbarlabels = cb.ax.get_yticklabels() 
        [label.set_fontname('Times New Roman') for label in cbarlabels]

    arrow_x1 = np.linspace(-1.8, 1.8, 3)
    arrow_x2 = np.linspace(-1.8, 1.8, 3)
    ax1_grid, ax2_grid = np.meshgrid(arrow_x1, arrow_x2)

    thetas = theta * np.ones_like(ax1_grid)
    ux = v * np.cos(thetas)
    uy = v * np.sin(thetas)
    ax.quiver(arrow_x1,arrow_x2,ux,uy,color='k',angles='xy', scale_units='xy', scale=2,alpha=0.5)
    return ax


def plot_pic(env, agent, model_location):

    fig, ([ax1,ax2,ax3,ax4]) = plt.subplots(
        nrows=1, ncols=4,
        figsize=(10.5, 2.5),
        constrained_layout=True,
    )
    
    my_x_ticks = np.arange(-3,3.01,1.5)
    my_y_ticks = np.arange(-3,3.01,1.5)

    labels = ax1.get_xticklabels() + ax1.get_yticklabels() + ax2.get_xticklabels() + ax2.get_yticklabels() \
    + ax3.get_xticklabels() + ax3.get_yticklabels() + ax4.get_xticklabels() + ax4.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


    '''
    subplot1 : plot the task
    '''
    ax1 = env.plot_task(ax1)
    ax1.set_xticks(my_x_ticks)
    ax1.set_yticks(my_y_ticks)
    ax1.set_xlim((-3, 3))  
    ax1.set_ylim((-3, 3))  
    ax1.tick_params(labelsize=ticks_size)
    ax1.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax1.spines['top'].set_color('gray')      
    ax1.spines['bottom'].set_color('gray')  
    ax1.spines['left'].set_color('gray')   
    ax1.spines['right'].set_color('gray')  

    
    '''
    subplot2,3,4 : plot the feasible region and the learned feasible region for different v and theta
    '''
    ax2 = plot_pr_pic(ax2, agent, v=0.5, theta=np.pi / 4)
    ax2 = env.plot_map(ax2, v=0.5, theta=np.pi / 4)

    ax3 = plot_pr_pic(ax3, agent, v=1, theta=np.pi / 2)
    ax3 = env.plot_map(ax3, v=1, theta=np.pi / 2)

    ax4 = plot_pr_pic(ax4, agent, v=1.5, theta=np.pi / 4, cb =True)
    ax4 = env.plot_map(ax4, v=1.5, theta=np.pi / 4)

    
    for ax in [ax2, ax3, ax4]:
        ax.set_xticks(my_x_ticks)
        ax.set_yticks(my_y_ticks)
        ax.set_xlim((-3, 3))
        ax.set_ylim((-3, 3))
        ax.tick_params(labelsize=ticks_size)
        ax.set_xlim([-2.7,2.7])
        ax.set_ylim([-2.7,2.7])
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.spines['bottom'].set_linewidth(width)
        ax.spines['left'].set_linewidth(width)
        ax.spines['right'].set_linewidth(width)
        ax.spines['top'].set_linewidth(width)
        ax.spines['top'].set_color('white') 
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white') 
        ax.spines['right'].set_color('white')
    
    plt.savefig(f"{model_location}/imgs/viz_map.png", dpi=600)


def load_diffusion_model(model_location):

    cfg = FLAGS.config

    env = eval('PointRobot')(id=0, seed=0)

    config_dict = dict(cfg['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    agent = globals()[model_cls].create(
        cfg['seed'], env.observation_space, env.action_space, **config_dict
    )

    def get_model_file():
        files = os.listdir(f"{model_location}")
        pickle_files = []
        for file in files:
            if file.endswith('.pickle'):
                pickle_files.append(file)
        numbers = {}
        for file in pickle_files:
            match = re.search(r'\d+', file)
            number = int(match.group())
            path = os.path.join(f"{model_location}", file)
            numbers[number] = path

        max_number = max(numbers.keys())
        max_path = numbers[max_number]
        return max_path
    
    model_file = get_model_file()
    new_agent = agent.load(model_file)

    if not os.path.exists(f"{model_location}/imgs"):
        os.makedirs(f"{model_location}/imgs")

    return env, new_agent

def main(_):

    diffusion_model_location = 'results/PointRobot/ddpm_feasibility_hj_N16_minqc_2023-10-29_208' # expert_random
    env, diffusion_agent = load_diffusion_model(diffusion_model_location)
    
    plot_pic(env, diffusion_agent, diffusion_model_location)


if __name__ == '__main__':
    app.run(main)
