export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu

# python launcher/examples/train_offline.py --env_id 0 --config configs/train_config.py:fisor_cb1

# python launcher/examples/train_offline.py --env_id 0 --config configs/train_config.py:fisor_cb1
# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:fisor --ratio 0.1
# python launcher/examples/train_offline.py --env_id 21 --config configs/train_config.py:fisor --ratio 0.1
# 

python launcher/examples/train_offline.py --env_id 29 --config configs/train_config.py:fisor