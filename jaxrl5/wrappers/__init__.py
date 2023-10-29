import gym
from gym.wrappers.flatten_observation import FlattenObservation
from jaxrl5.wrappers.single_precision import SinglePrecision


def wrap_gym(env: gym.Env, rescale_actions: bool = True, cost_limit: int = 1) -> gym.Env:
    env = SinglePrecision(env)

    if rescale_actions:
        env = gym.wrappers.RescaleAction(env, -1, 1)

    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env = gym.wrappers.ClipAction(env)
    env.set_target_cost(cost_limit)
    print('env_cost_limit', env.target_cost)
    return env
