"""
Does not work. Gives the following error -
AttributeError: module '_Box2D' has no attribute 'RAND_LIMIT_swigconstant'

According to Box2D forums this is a regression in swig and the bug is still open.
"""

import gym
from run_env import run

env = gym.make('LunarLander-v2')
run(env)
