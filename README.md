# IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies

Paper Link : https://arxiv.org/abs/2304.10573

Check out https://github.com/philippe-eecs/JaxDDPM for an implementation of DDPMs in JAX for continuous spaces!

# Reproducing Results

[Offline Script Location.](launcher/examples/train_ddpm_iql_offline.py)

Run Line for each variant. Edit the script location above to change hyperparameters and environments to sweep over. 

```
python3 launcher/examples/train_ddpm_iql_offline.py --variant 0...N
```

[Finetune Script Location.](launcher/examples/train_ddpm_iql_finetune.py)

Run 
```
python3 launcher/examples/train_ddpm_iql_finetune.py --variant 0...N
```

# Important File Locations

[Main run script were variant dictionary is passed.](/examples/states/train_diffusion_offline.py)

[DDPM Implementation.](/jaxrl5/networks/diffusion.py)

[LN_Resnet.](/jaxrl5/networks/resnet.py)

[DDPM IQL Learner.](/jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/ikostrikov/jaxrl5/tree/main.svg?style=svg&circle-token=668374ebe0f27c7ee70edbdfbbd1dd928725c01a)](https://dl.circleci.com/status-badge/redirect/gh/ikostrikov/jaxrl5/tree/main) [![codecov](https://codecov.io/gh/ikostrikov/jaxrl5/branch/main/graph/badge.svg?token=Q5QMIDZNZ3)](https://codecov.io/gh/ikostrikov/jaxrl5)

# Installation

Run
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

Based from a re-implementation of https://github.com/ikostrikov/jaxrl 

# Citations
Cite this paper
```
@misc{hansenestruch2023idql,
      title={IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies}, 
      author={Philippe Hansen-Estruch and Ilya Kostrikov and Michael Janner and Jakub Grudzien Kuba and Sergey Levine},
      year={2023},
      eprint={2304.10573},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

Please also cite the JAXRL repo as well if you use this repo
```
@misc{jaxrl,
  author = {Kostrikov, Ilya},
  doi = {10.5281/zenodo.5535154},
  month = {10},
  title = {{JAXRL: Implementations of Reinforcement Learning algorithms in JAX}},
  url = {https://github.com/ikostrikov/jaxrl},
  year = {2021}
}
```

python launcher/viz/viz_map.py --model_location 'results/PointRobot/ddpm_feasibility_hj_N16_minqc_2023-12-09_s54_486'
