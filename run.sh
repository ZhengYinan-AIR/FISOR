export XLA_PYTHON_CLIENT_PREALLOCATE=False
export CUDA_VISIBLE_DEVICES=0
# export JAX_PLATFORM_NAME=cpu

# metadrive
python launcher/examples/train_offline.py --env_id 6 --config configs/train_config.py:fisor --ratio 0.01
# python launcher/examples/train_offline.py --env_id 6 --config configs/train_config.py:fisor --ratio 0.5
# python launcher/examples/train_offline.py --env_id 6 --config configs/train_config.py:fisor --ratio 0.5
# # imitation
# python launcher/examples/train_offline.py --env_id 3 --config configs/train_config.py:fisor_imitation