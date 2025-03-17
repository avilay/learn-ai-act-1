def run(env):
    for episode in range(20):
        print('Starting new episode')
        obs = env.reset()
        for t in range(100):
            env.render()
            print(obs)
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                print('Episode finished after {} timesteps'.format(t+1))
                break
