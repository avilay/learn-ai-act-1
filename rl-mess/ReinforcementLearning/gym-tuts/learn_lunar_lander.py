import sys
import gym
import gym.spaces


env = gym.make("LunarLander-v2")
if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
    help(env.env)
    sys.exit(0)

while True:
    try:
        observation = env.reset()
        tot_reward = 0
        done = False
        while not done:
            env.render()
            # action = int(input("\nAction (0-5)? "))
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            tot_reward += reward
    except KeyboardInterrupt:
        sys.exit(0)

