import gym
import gym.spaces

from dqn_agent import Agent

# watch an untrained agent
env = gym.make('LunarLander-v2')
env.seed(0)
agent = Agent(state_size=8, action_size=4, seed=0)

ans = "y"
while ans == "y":
    state = env.reset()
    tot_reward = 0
    for j in range(750):
        action = agent.act(state, eps=0.9)
        env.render()
        state, reward, done, _ = env.step(action)
        tot_reward += reward
        if done:
            print(f"Game End - Score: {tot_reward}")
            break
    ans = input("Continue(Y/N)? ")

env.close()
