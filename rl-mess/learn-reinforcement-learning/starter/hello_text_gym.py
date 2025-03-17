import sys
import time

import gym
import numpy as np

envname = sys.argv[1]
env = gym.make(envname)
num_episodes = 1
max_t = 1000
sleep_time_secs = 1

for episode in range(num_episodes):
    print(f"\nStarting episode {episode}")
    state = env.reset()
    print(state)
    try:
        env.render()
    except NotImplementedError:
        pass
    time.sleep(sleep_time_secs)
    t = 0
    done = False
    rewards = []
    while not done and t < max_t:
        action = env.action_space.sample()
        print("Action:", action)
        state, reward, done, _ = env.step(action)
        print(f"State: {state}\tReward: {reward}")
        try:
            env.render()
        except NotImplementedError:
            pass
        time.sleep(sleep_time_secs)
        rewards.append(reward)
        t += 1
    print(f"Episode {episode} took {t} timesteps with an average reward of {np.mean(rewards)}")

env.close()
