import sys
import time

import gym
import numpy as np

envname = sys.argv[1]
env = gym.make(envname)
another_episode = True
max_t = 1000
episode = 0
while another_episode:
    print(f"\nStarting episode {episode}")
    state = env.reset()
    env.render()
    t = 0
    done = False
    rewards = []
    while not done and t < max_t:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        env.render()
        rewards.append(reward)
        t += 1
    print(f"Episode {episode} took {t} timesteps with an average reward of {np.mean(rewards):.2f}")
    ans = input("Another episode (Y/n)? ")
    another_episode = False if ans == "n" else True
    episode += 1
env.close()
