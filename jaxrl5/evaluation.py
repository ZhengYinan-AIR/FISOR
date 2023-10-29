from typing import Dict

import gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}
