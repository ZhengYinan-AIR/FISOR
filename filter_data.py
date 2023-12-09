import gym
import numpy as np
import dsrl
from collections import defaultdict
import h5py
from tqdm.auto import trange  # noqa

# Use for Saute RL dataset
def state_augmentation(dataset_dict, cost_limit):
    data_num = dataset_dict['observations'].shape[0]
    
    observations = []
    next_observations = []
    
    is_start = True
    for i in trange(data_num, desc='data_processing'):
        if is_start:
            safe_state = 1.0
            is_start = False

        observations.append(np.hstack([dataset_dict['observations'][i], safe_state]))
        safe_state -= dataset_dict['costs'][i] / cost_limit
        safe_state /= 0.99
        next_observations.append(np.hstack([dataset_dict['next_observations'][i], safe_state]))

        if safe_state <= 0:
            dataset_dict['rewards'][i] = -10

        if dataset_dict['terminals'][i] or dataset_dict['timeouts'][i]:
            is_start = True

    print(len(observations))
    print(len(next_observations))

    observations = np.array(observations)
    next_observations = np.array(next_observations)

    print(observations.shape)
    print(next_observations.shape)

    keys = [
        'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]

    output_path = 'OfflineCarPush1Gymnasium-v0-10.hdf5'
    outf = h5py.File(output_path, 'w')
    for k in keys:
        outf.create_dataset(k, data=dataset_dict[k], compression='gzip')
    outf.create_dataset('observations', data = observations, compression='gzip')
    outf.create_dataset('next_observations', data=next_observations, compression='gzip')
    outf.close()


    return dataset_dict

# Use for data quntity exp
def filter_dataset(data_dict, ratio):
    done_idx = np.where(
        (data_dict["terminals"] == 1) | (data_dict["timeouts"] == 1)
    )[0]

    trajs= []
    for i in range(done_idx.shape[0]):
        start = 0 if i == 0 else done_idx[i - 1] + 1
        end = done_idx[i] + 1
        traj = {k: data_dict[k][start:end] for k in data_dict.keys()}
        trajs.append(traj)

    print(
        f"before filter: traj num = {len(trajs)}, transitions num = {data_dict['observations'].shape[0]}"
    )

    traj_idx = np.random.randint(0, len(trajs), size=int(len(trajs) * ratio))

    processed_data_dict = defaultdict(list)
    for k in data_dict.keys():
        for i in traj_idx:
            processed_data_dict[k].append(trajs[i][k])
    processed_data_dict = {
        k: np.concatenate(v)
        for k, v in processed_data_dict.items()
    }
    

    print(
        f"before filter: traj num = {traj_idx.shape[0]}, transitions num = {processed_data_dict['observations'].shape[0]}"
    )

    keys = [
        'observations', 'next_observations', 'actions', 'rewards', 'costs', 'terminals',
        'timeouts'
    ]

    output_path = 'data/SafeMetaDrive-hardsparse-v0-85-' + str(traj_idx.shape[0]) + '-' + str(ratio) + '.hdf5'
    outf = h5py.File(output_path, 'w')
    for k in keys:
        outf.create_dataset(k, data=processed_data_dict[k], compression='gzip')
    outf.close()


env = gym.make("OfflineMetadrive-hardsparse-v0")
dataset_dict = env.get_dataset()

for k, v in dataset_dict.items():
    dataset_dict[k] = v.astype(np.float32)


dataset_dict = filter_dataset(dataset_dict, 0.01)






