import sys
import gym
import gym.spaces


env = gym.make("Taxi-v2")
if len(sys.argv) > 1 and (sys.argv[1] == "--help" or sys.argv[1] == "-h"):
    help(env.env)
    sys.exit(0)

observation = env.reset()
env.action_space
tot_reward = 0
for _ in range(1000):
    env.render()
    action = int(input("\nAction (0-5)? "))
    observation, reward, done, info = env.step(action)
    tot_reward += reward
    print("Reward: ", tot_reward)
    if done:
        print("\n***Game End***\n")
        sys.exit(0)

