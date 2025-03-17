"""TD Learning.

Usage:
  td_learning.py (CliffWalking-v0 | FrozenLake-v0 [--slippery]) (q | sarsa) [--eps=<eps>] [--alpha=<alpha>] [--max_t=max_t] [--gamma=gamma]
  td_learning.py (-h | --help)
  td_learning.py (-v | --version)

Arguments:
  <env>             The environment to run. Valid values are cliff_walking, frozen_lake, frozen_lake_no_slip.

Options:
  -h --help         Show this screen.
  --version         Show version.
  --eps=<eps>       Epsilong greedy parameter [default: 1.0].
  --alpha=<alpha>   Learning rate [default: 0.1].
  --max_t=<max_t>   Maxinumum number of timesteps [default: 1000].
  --gamma=<gamma>   Reward discount factor [default: 0.999].

Both Q-learning and SARSA are iterative convergence algorithm. The update rules are slightly different.
Update rule for Q-learning:
Q(s, a) <- Q(s, a) + α[r + γ maxQ(s', a') - Q(s, a)]

Update rule for SARSA:
Q(s, a) <- Q(s, a) + α[r + γQ(s', a') - Q(s, a)]

Optimal hyperparams for CliffWalking-v0 with Q-Learning are:
eps = 1. [default]
alpha = 0.1 [default]
max_t = 100,000
gamma = 0.999 [default]

Optimal hyperparams for CliffWalking-v0 with SARSA are:
eps = 1.0 [default]
alpha = 0.01
max_t = 150000
gamma=0.999 [default]

Optimal hyperparams for FrozenLake-v0 (not slippery) with Q-Learning are:
eps = 1. [default]
alpha = 0.1 [default]
max_t = 10,000
gamma = 0.999 [default]

Optimal hyperparams for FrozenLake-v0 (not slippery) with SARSA are:
eps = 1.0 [default]
alpha = 0.05
max_t = 15000
gamma = 0.999 [default]
"""


import logging
import random
import sys

import gym
import numpy as np
from bokeh.io import show
from bokeh.layouts import row
from bokeh.plotting import figure
from docopt import docopt

UP = " ↑ "
RIGHT = " → "
DOWN = " ↓ "
LEFT = " ← "


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


# Takes greedy action with prob 1-eps
def eps_greedy(eps_0, t, Q, state):
    # Anneal eps every 1000 steps
    k = (t // 1000) + 1
    eps = eps_0 / k

    if random.random() > eps:
        return np.argmax(Q[state])
    else:
        n = Q.shape[-1]
        return random.choice(np.arange(n))


def td_learning(algo, env, alpha, max_t, eps_0, gamma):
    m = env.observation_space.n
    n = env.action_space.n
    Q = np.zeros((m, n))

    # There are two evaluation metrics -
    #   - the score in an episode; this should increase with training
    #   - the number of steps taken to complete an episode; this should decrease with training
    scores = []
    episode_steps = []

    t = 0
    while t < max_t:
        # Start a new episode
        logging.debug("Starting a new episode")
        score = 0
        done = False
        state = env.reset()
        start_t = t
        while not done:
            action = eps_greedy(eps_0, t, Q, state)
            next_state, reward, done, _ = env.step(action)
            t += 1
            if algo == "q_learning":
                td_target = reward + gamma * np.max(Q[next_state]) if not done else reward
            elif algo == "sarsa":
                next_action = eps_greedy(eps_0, t, Q, next_state)
                td_target = reward + gamma * Q[next_state, next_action] if not done else reward
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            score += reward
            state = next_state if not done else None
        scores.append(score)
        episode_steps.append(t - start_t + 1)
        logging.debug(
            f"Episode ended after {episode_steps[-1]} timesteps and a score of {score}.\n"
        )
    logging.info(f"Learnt from {len(scores)} episodes.")
    return Q, scores, episode_steps


def main():
    args = docopt(__doc__, version="TD Learning 1.0")

    # Get the env
    if args["CliffWalking-v0"]:
        env = gym.make("CliffWalking-v0")
        labels = [UP, RIGHT, DOWN, LEFT]
        ncols = 12
    elif args["FrozenLake-v0"]:
        labels = [LEFT, DOWN, RIGHT, UP]
        ncols = 4
        if args["--slippery"]:
            env = gym.make("FrozenLake-v0")
        else:
            env = gym.make("FrozenLake-v0", is_slippery=False)

    # Get the algo
    if args["q"]:
        algo = "q_learning"
    elif args["sarsa"]:
        algo = "sarsa"

    # Get the hparams
    eps = float(args["--eps"])
    alpha = float(args["--alpha"])
    max_t = int(args["--max_t"])
    gamma = float(args["--gamma"])

    print(
        f"Starting {algo.upper()} learning with eps={eps}, alpha={alpha}, max_t={max_t}, and gamma={gamma}"
    )
    env.render()

    Q, scores, episode_steps = td_learning(algo, env, alpha, max_t, eps, gamma)
    print_policy(Q, labels, ncols)
    print("Last 10 scores: ", scores[-10:])
    p1 = plot(scores, "Scores")
    p2 = plot(episode_steps, "Steps/Episode")
    show(row(p1, p2))


logging.basicConfig(level=logging.DEBUG, filename="td.log")
if __name__ == "__main__":
    main()
