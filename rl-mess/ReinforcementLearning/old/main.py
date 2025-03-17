import gym
import gym.spaces
import gymrl.temporal_difference.sarsa as sarsa
from gymrl.common import gen_greedy_policy

env = gym.make('CliffWalking-v0')
Q = sarsa.learn(env, 10, 0.01)
policy = gen_greedy_policy(Q)
print(policy)
