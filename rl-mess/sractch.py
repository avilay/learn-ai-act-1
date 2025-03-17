import gym
import numpy as np


def main():
    env = gym.make("CarRacing-v0")
    cont = True
    s = env.reset()
    env.render()
    while cont:
        a = env.action_space.sample()
        s, r, done, _ = env.step(a)
        env.render()
        if done:
            cont = bool(input("Episode ended. Continue: "))
            if cont:
                s = env.reset()


main()
