"""
Gym is incompatible w ith the latest mujoco-py. They claim that it works with versio 0.5 which is not
available on pypi. The support for the latest version is on the roadmap.
https://github.com/openai/mujoco-py/issues/80
"""
import gym
from run_env import run

env = gym.make('Humanoid-v1')
run(env)
