from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='FISOR-metadrive',
        seed=1,
        max_steps=1000001,
        eval_episodes=20,
        batch_size=2048, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1000,
        eval_interval=250000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
    )

    possible_structures = {
        "fisor": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="FISOR",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                    beta_schedule='vp',
                    actor_objective="feasibility", 
                    sampling_method="ddpm", 
                    extract_method="minqc", 
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "fisor_imitation": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="FISOR",
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cost_temperature=5,
                    reward_temperature=3,
                    T=5,
                    N=16,
                    M=0,
                    clip_sampler=True,
                    actor_dropout_rate=0.1,
                    actor_num_blocks=3,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    actor_layer_norm=True,
                    value_layer_norm=False,
                    actor_tau=0.001,
                    actor_architecture='ln_resnet',
                    critic_objective='expectile',
                    critic_hyperparam = 0.9,
                    cost_critic_hyperparam = 0.9,
                    critic_type="hj", #[hj, qc]
                    cost_ub=150,
                    beta_schedule='vp',
                    actor_objective="imitation", 
                    sampling_method="ddpm", 
                    extract_method="minqc", 
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]