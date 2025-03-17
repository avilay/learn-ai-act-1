import numpy as np

STICK = 0
HIT = 1


class Step:
    def __init__(self, **kwargs):
        self.state = kwargs.get('state')
        self.action = kwargs.get('action')
        self.reward = kwargs.get('reward')

    def __repr__(self):
        action = 'STICK' if self.action == STICK else 'HIT'
        return f'<Step(state={self.state} action={action} reward={self.reward})>'


def calc_cum_rewards(episode, γ):
    cum_rewards = [None] * len(episode)
    cum_rewards[-1] = episode[-1].reward
    # Now start from the second last element and go backwards
    for i_step in range(len(episode)-2, -1, -1):
        cum_reward = episode[i_step].reward + γ*cum_rewards[i_step+1]
        cum_rewards[i_step] = cum_reward
    return cum_rewards

def gen_episode(env, policy):
    episode = []
    done = False
    state = env.reset()
    while not done:
        pmf = [None] * env.action_space.n
        for action in range(env.action_space.n):
            pmf[action] = policy(state, action)
        action = np.random.choice(list(range(len(pmf))), p=pmf)
        next_state, reward, done, info = env.step(action)
        step = Step(state=state, action=action, reward=reward)
        episode.append(step)
        state = next_state
    return episode
