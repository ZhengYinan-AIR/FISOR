import os
import sys
sys.path.append('.')
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags
import wandb
from tqdm.auto import trange  # noqa
import gymnasium as gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import FISOR
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr


FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_string('experiment_name', '', 'experiment name for wandb')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)


def call_main(details):
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'])
    wandb.config.update(details)

    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name'])
        ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    config_dict['env_max_steps'] = env_max_steps

    model_cls = config_dict.pop("model_cls") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )


    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])
        agent, info = agent.update(sample)
        
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)

        # if i % details['eval_interval'] == 0 and i > 0:
        if i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['group']}/{details['experiment_name']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            if details['env_name'] != 'PointRobot':
                eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)


def main(_):
    parameters = FLAGS.config
    np.random.seed(parameters['seed'])

    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['group'] = parameters['env_name']

    parameters['experiment_name'] = parameters['agent_kwargs']['sampling_method'] + '_' \
                                + parameters['agent_kwargs']['actor_objective'] + '_' \
                                + parameters['agent_kwargs']['critic_type'] + '_N' \
                                + str(parameters['agent_kwargs']['N']) + '_' \
                                + parameters['agent_kwargs']['extract_method'] if FLAGS.experiment_name == '' else FLAGS.experiment_name
    parameters['experiment_name'] += '_' + str(datetime.date.today()) + '_' + str(parameters['seed']) 

    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['cost_temperature'] = 2
        parameters['agent_kwargs']['reward_temperature'] = 5
        parameters['agent_kwargs']['cost_ub'] = 150
        parameters['agent_kwargs']['N'] = 8

    print(parameters)

    if not os.path.exists(f"./results/{parameters['group']}/{parameters['experiment_name']}"):
        os.makedirs(f"./results/{parameters['group']}/{parameters['experiment_name']}")
    with open(f"./results/{parameters['group']}/{parameters['experiment_name']}/config.yaml", "w") as f:
        yaml.dump(dict(parameters), f, default_flow_style=False, allow_unicode=True)
    
    call_main(parameters)


if __name__ == '__main__':
    app.run(main)
