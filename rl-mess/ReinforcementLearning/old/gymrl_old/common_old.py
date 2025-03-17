from collections import defaultdict
from typing import Mapping, Any, Dict, Sequence, List, Optional
import numpy as np
import gym

# A function that takes in the state and optionally the action as input
# and outputs the π(a|s). If the action is not given, it outputs the entire probability
# distribution for the state.
# typing has no way of specifying functions with optional arguments :-(
PolicyFunction = Any


class Step:
    """Represents a single time step in an episode
    Consider an episode like -
    S0, A0, R1, S1, A1, R2, S2, A2, ..., Rt, St, At, R{t+1}, ...
    Where an agent starts out in state S0, takes an action A0, gets a reward R1 and lands in state
    S1 and so on.
    Then the step object represents -
    [S0, A0], [R1, S1, A1], [R2, S2, A2], ..., [Rt, St, At], ...

    Attributes:
        action_labels: This is a class-level attribute to give meaningful names to actions.

        reward: The reward for the action taken in the previous time step.
        state: The state in which the agent finds itself in the current time step.
        action: The action that the agent takes in the current time step.
    """
    action_labels: List[str]

    def __init__(self, *,
                 reward: Optional[float]=None,
                 state: Optional[Any]=None,
                 action: Optional[int]=None) -> None:
        self.reward = reward
        self.state = state
        self.action = action

    def __repr__(self):
        action_label: str = Step.action_labels[self.action] if self.action else 'NONE'
        return f'<Step(reward={self.reward} state={self.state} action={action_label})>'


def gen_epsilon_greedy_policy(num_actions, Q: Mapping[Any, Sequence[float]], ε: float) -> PolicyFunction:
    """A higher-order function that generates the epsilon-greedy policy as a function

    Args:
        Q: Can be conceptually thought of as a table of values with states as rows and actions as
        columns. It is implemented as a dict which has states as keys (because states can be any
        object) and a numpy array of floats. The index of the numpy array is the action and its
        value is the action-value for that state.

        ε: A value between [0, 1] indicating the exploration-exploitation trade-off. Higher values
        mean more exploration.

    Returns:
        A policy function that takes the state and action as input and outputs π(a|s). For each
        state, the best action has a probability of 1 - ε + ε/len(A) and and all other
        actions have a probability of ε/len(A). If the client does not specify the action then
        it returns the probability distribution across all actions.
    """
    pmf: Dict[Any, np.ndarray] = defaultdict(lambda: np.full(num_actions, 1/num_actions))
    for state, action_values in Q.items():
        base_prob = ε / num_actions
        pmf[state] = np.full(num_actions, base_prob)

        # Get the best value in this state. It is possible that it is a dup, i.e., multiple actions
        # have the best value. In that case, choose one of these actions at random and call it the
        # best action.
        best_value = np.max(action_values)
        best_actions = np.argwhere(action_values == best_value).flatten()
        best_action = np.random.choice(best_actions)
        # Now this action has a probability of 1 - ε + ε/len(A). All other actions
        # have probability ε/len(A).
        pmf[state][best_action] += (1 - ε)

    def policy(state: Any, action: Optional[int]=None) -> float:
        return pmf[state][action] if action is not None else pmf[state]

    return policy


def gen_episode(env: gym.Env, policy: PolicyFunction) -> Sequence[Step]:
    """Generates an episode for environments with terminal states

    Args:
        env: The gym environment.
        policy: The policy to use to traverse the environment

    Return:
        A sequence of steps with the index indicating time.
    """
    episode: List[Step] = []
    done = False
    state = env.reset()
    step = Step(state=state)
    episode.append(step)
    while not done:
        step = episode[-1]
        pmf = policy(step.state)
        step.action = np.random.choice(np.arange(len(pmf)), p=pmf)
        next_state, reward, done, info = env.step(step.action)
        next_step = Step(state=next_state, reward=reward)
        episode.append(next_step)

    return episode


def gen_greedy_policy(Q: Mapping[Any, Sequence[float]]) -> PolicyFunction:
    """A policy that gives the best (greedy) action for any state

    Args:
        Q: Can be conceptually thought of as a table of values with states as rows and actions as
        columns. It is implemented as a dict which has states as keys (because states can be any
        object) and a numpy array of floats. The index of the numpy array is the action and its
            value is the action-value for that state.

    Returns:
        A mapping that has the env state as its key and the greedy action as its value.
    """
    policy: Dict[Any, int] = {}
    for state, action_values in Q.items():
        # Get the best value in this state. It is possible that it is a dup, i.e., multiple actions
        # have the best value. In that case, choose one of these actions at random and call it the
        # best action.
        best_value = np.max(action_values)
        best_actions = np.argwhere(action_values == best_value).flatten()
        best_action = np.random.choice(best_actions)

        def policy(state: Any, action: Optional[int]=None) -> int:
        policy[state] = best_action
    return policy


def calc_cum_rewards(episode: Sequence[Step], γ: Optional[float]=1.) -> Sequence[float]:
    """Calculates the discounted reward at each time step
    The general formula is -
    $ G_t = R_{t+1} + γG_{t+1} $
    This means that for the last time step n: G_n = R_{n+1} + γG_{n+1} = None or 0.
    $ G_{n-1} = R_{n} + γG_{n} = R_{n} $
    And finally $ G_0 = R_1 + γG_1  $
    Of course R_0 is None or 0 because R_t represents the reward for action taken in the last
    time step.

    Args:
        episode: A sequence of steps where the index represents time.
        γ: The discount factor. Higher number means more emphasis on future rewards.

    Returns:
        G: A sequence of discounted rewards where the index reprsents time.
    """
    cum_rewards = np.zeros(len(episode))
    cum_rewards[-1] = 0
    # Now start from the second last element and go backwards
    for i_step in range(len(episode)-2, -1, -1):
        cum_rewards[i_step] = episode[i_step+1].reward + γ * cum_rewards[i_step+1]
    return cum_rewards
