import logging
import pickle
import random
from dataclasses import dataclass

import gym
import numpy as np
from bokeh.io import show
from bokeh.layouts import row
from bokeh.plotting import figure

UP = " ↑ "
RIGHT = " → "
DOWN = " ↓ "
LEFT = " ← "


@dataclass
class Step:
    s: int
    a: int
    r: float
    s_: int
    g: float


def print_policy(Q, labels, ncols):
    m = Q.shape[0]
    for state in range(m):
        action = np.argmax(Q[state])
        label = labels[action]
        print(label, end="")
        if (state + 1) % ncols == 0:
            print("\n")


def plot(vals, title):
    moving_avg = np.convolve(vals, np.ones(1000), "valid") / 1000
    p = figure(plot_height=400, plot_width=400, title=title)
    x = np.arange(len(moving_avg))
    p.line(x, moving_avg)
    p.xaxis.axis_label = "Episode"
    return p


def eps_greedy(Q, eps):
    def policy(state):
        if random.random() > eps:
            return np.argmax(Q[state])
        else:
            n = Q.shape[-1]
            return random.choice(np.arange(n))

    return policy


def play(env, policy, gamma):
    done = False
    steps = []
    s = env.reset()
    while not done:
        a = policy(s)
        s_, r, done, _ = env.step(a)
        steps.append(Step(s, a, r, s_, None))
        s = s_

    steps[-1].g = steps[-1].r
    for k in range(len(steps) - 2, -1, -1):
        steps[k].g = steps[k].r + gamma * steps[k + 1].g
    return steps


def main():
    num_episodes = 100000
    envname = "CliffWalking-v0"
    gamma = 1.0
    eps = 1.0
    k = 0.999
    alpha = 0.02
    env = gym.make(envname)
    m = env.observation_space.n
    n = env.action_space.n
    N = np.zeros((m, n))
    Q = np.random.standard_normal((m, n))
    scores = []
    episode_steps = []
    try:
        for i_episode in range(1, num_episodes + 1):
            # eps = 1 / i_episode
            eps = max(eps * k, 0.05)
            policy = eps_greedy(Q, eps)
            episode = play(env, policy, gamma)
            scores.append(episode[0].g)
            episode_steps.append(len(episode))
            for step in episode:
                s = step.s
                a = step.a
                g = step.g
                error = g - Q[s, a]
                # N[s, a] += 1
                # Q[s, a] += error / N[s, a]
                Q[s, a] += alpha * error
            logging.debug(
                f"Episode {i_episode} ended after {episode_steps[-1]} timesteps and a score of {scores[-1]}.\n"
            )
    except KeyboardInterrupt:
        print("User interrupted training!")

    env.render()
    labels = [UP, RIGHT, DOWN, LEFT]
    print_policy(Q, labels, 12)
    print("Last 10 scores: ", scores[-10:])
    p1 = plot(scores, "Scores")
    p2 = plot(episode_steps, "Steps/Episode")
    show(row(p1, p2))
    with open("Qvals.pkl", "wb") as f:
        pickle.dump(Q, f, pickle.HIGHEST_PROTOCOL)
    with open("Nvals.pkl", "wb") as f:
        pickle.dump(N, f, pickle.HIGHEST_PROTOCOL)


logging.basicConfig(level=logging.DEBUG, filename="mc.log")

if __name__ == "__main__":
    main()
