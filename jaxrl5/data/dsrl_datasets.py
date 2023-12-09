import os
import gym
import dsrl
import numpy as np
from jaxrl5.data.dataset import Dataset
import h5py


class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, critic_type="qc", data_location=None, cost_scale=1., ratio = 1.0):
        # imitation
        if data_location is not None:
            dataset_dict = env.get_dataset(h5path=data_location)
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost, 
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                                dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict['timeouts']
        else:
            # DSRL
            if ratio == 1.0:
                dataset_dict = env.get_dataset()
            else:
                _, dataset_name = os.path.split(env.dataset_url)
                file_list = dataset_name.split('-')
                ratio_num = int(float(file_list[-1].split('.')[0]) * ratio)
                dataset_ratio = '-'.join(file_list[:-1]) + '-' + str(ratio_num) + '-' + str(ratio) + '.hdf5'
                dataset_dict = env.get_dataset(os.path.join('data', dataset_ratio))
            print('max_episode_reward', env.max_episode_reward, 
                'min_episode_reward', env.min_episode_reward,
                'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('max_episode_cost', env.max_episode_cost, 
                'min_episode_cost', env.min_episode_cost,
                'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
            dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                                dataset_dict["timeouts"]).astype(np.float32)
            del dataset_dict["terminals"]
            del dataset_dict['timeouts']

            
            if critic_type == "hj":
                dataset_dict['costs'] = np.where(dataset_dict['costs']>0, 1*cost_scale, -1)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        super().__init__(dataset_dict)
