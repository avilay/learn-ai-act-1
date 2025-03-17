import gym

env = gym.make("SpaceInvaders-v0")
for _ in range(1000):
    state = env.reset()
    tot_reward = 0
    done = False
    while not done:
        env.render()
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        tot_reward += reward
    print(f"Episoded ended with total reward {tot_reward}")
