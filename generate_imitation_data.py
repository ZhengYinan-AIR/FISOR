import gym
import dsrl
from collections import defaultdict
import os
import os.path as osp
from absl import app, flags
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

from env.utils import get_trajectory_info, filter_trajectory
from env.env_list import env_list


label_size = 15
ticks_size = 15
legend_size = 10
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : label_size,
}
color = ['#0d5b26', '#F3BE69']

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 3, 'Choose env')
flags.DEFINE_integer('reward_min', 220, 'Expert reward')

def main(_):
    assert FLAGS.env_id >= 3 and FLAGS.env_id <=5 # only medium env
    env_name = env_list[FLAGS.env_id]
    env = gym.make(env_name)
    dataset_dict = env.get_dataset()

    if not os.path.exists("data"):
        os.makedirs("data")

    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]

    # process the dataset to the SDT format:
    # traj[i] is the i-th trajectory (a dict)
    rew_ret, cost_ret, start_index, end_index = get_trajectory_info(dataset_dict)
    traj = []
    for i in trange(len(rew_ret), desc="Processing trajectories..."):
        start = start_index[i]
        end = end_index[i] + 1
        one_traj = {k: dataset_dict[k][start:end] for k in keys}
        traj.append(one_traj)
    print(f"Total number of trajectories: {len(traj)}")

    # plot the original dataset cost-reward figure
    fig = plt.figure(figsize=(5.5,5))

    fig1 = fig.add_subplot(1,1,1) 
    plt.scatter(cost_ret, rew_ret,s=15,color=color[0])

    reward_min = FLAGS.reward_min
    # downsampling the trajectories by grid filter
    cost_ret, rew_ret, traj = filter_trajectory(
        cost_ret,
        rew_ret,
        traj,
        rew_min=reward_min,
        rew_max=np.inf,
    )

    print(f"Num of trajectories after filtering: {len(traj)}")

    # plot the filtered dataset cost-reward figure
    plt.scatter(cost_ret, rew_ret,s=15,color=color[1])
    plt.xlabel("Cost",fontdict=font)
    plt.ylabel("Return",fontdict=font)
    plt.title(env_name,fontdict=font)

    plt.tick_params(labelsize=ticks_size)
    labels = fig1.get_xticklabels() + fig1.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    output_name = "data/" + env_name + "-"+ str(reward_min)
    output_path = output_name + '.png'
    plt.savefig(output_path, dpi=300)

    # process the trajectory data back to the d4rl data format:

    dataset = defaultdict(list)
    for d in traj:
        for k in keys:
            dataset[k].append(d[k])
    for k in keys:
        dataset[k] = np.squeeze(np.concatenate(dataset[k], axis=0))
        print(k, np.array(dataset[k]).shape, dataset[k].dtype)

    output_path = output_name + '.hdf5'
    outf = h5py.File(output_path, 'w')
    for k in keys:
        outf.create_dataset(k, data=dataset[k], compression='gzip')
    outf.close()

if __name__ == '__main__':
    app.run(main)
