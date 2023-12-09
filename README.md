# FISOR in MetaDrive

## Environment Installation
Install the ``MetaDrive`` environment via
```
pip install git+https://github.com/HenryLHH/metadrive_clean.git@main
```

## Main results
Run
``` Bash
# OfflineMetadrive-easysparse-v0
export XLA_PYTHON_CLIENT_PREALLOCATE=False
python launcher/examples/train_offline.py --env_id 0 --config configs/train_config.py:fisor
```
where ``env_id`` serves as an index for the [list of environments](https://github.com/ZhengYinan-AIR/FISOR/tree/metadrive_imitation/env/env_list.py).

## Data Quantity Experiments
The usage is the same as the [master](https://github.com/ZhengYinan-AIR/FISOR) branch.

## Imitation Learning
We can run [generate_data.sh](https://github.com/ZhengYinan-AIR/FISOR/blob/metadrive_imitation/generate_data.sh) to generate offline data of imitation learning. We also can download the necessary offline datasets ([Download link](https://cloud.tsinghua.edu.cn/d/12e08241957648d98493/)). Then run
``` Bash
python launcher/examples/train_offline.py --env_id 3 --config configs/train_config.py:fisor_imitation
```