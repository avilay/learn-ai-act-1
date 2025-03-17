from collections import defaultdict
from typing import List, Dict, Tuple
from rltk.core import Env, Policy, Agent, Episode, StateValues, State, ActionValues, Action

def mc_learning(env: Env, pi: Policy, num_iters=1000) -> StateValues:
    # Generate episodes
    agent = Agent(env)
    episodes: List[Episode] = []
    for _ in range(num_iters):
        episode = agent.move_to_end(pi)
        episodes.append(episode)

    cum_svals: Dict[State, float] = defaultdict(float)
    n: Dict[State, int] = defaultdict(int)
    for episode in episodes:
        episode.calc_cum_rewards()
        for step in episode:
            t, s = step.t, step.s
            cum_svals[s] += episode.cum_reward(t)
            n[s] += 1

    v: StateValues = StateValues.zeros()
    for s, cum_sval in cum_svals.items():
        v[s] = cum_sval/n[s]

    return v


def mc_incremental_learning(env: Env, pi: Policy, num_iters=1000) -> StateValues:
    agent = Agent(env)
    v: StateValues = StateValues.zeros()
    n: Dict[State, int] = defaultdict(int)
    for _ in range(num_iters):
        episode = agent.move_to_end(pi)
        episode.calc_cum_rewards()
        for step in episode:
            t, s = step.t, step.s
            g_t = episode.cum_reward(t)
            n[s] += 1
            v[s] = v[s] + (g_t - v[s]) / n[s]
    return v


def mc_control(env: Env, num_iters=1000) -> Policy:
    agent = Agent(env)
    q: ActionValues = ActionValues.random()
    pi: Policy = Policy.epsilon_greedy(q)
    n: Dict[Tuple[State, Action], int] = defaultdict(int)
    for k in range(num_iters):
        # Update values based on policy
        episode = agent.move_to_end(pi)
        episode.calc_cum_rewards()
        for step in episode:
            t, s, a = step.t, step.s, step.a
            g_t = episode.cum_reward(t)
            n[(s, a)] += 1
            q[s, a] = q[s, a] + (g_t - q[s, a])

        # Update policy based on values
        pi = Policy.epsilon_greedy(q, epsilon=1/k)
    optimal_pi = Policy.greedy(q)
    return optimal_pi
