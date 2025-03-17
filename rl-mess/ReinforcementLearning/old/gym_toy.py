import numpy as np
import gym
import gym.spaces
import mcq

STICK = 0
HIT = 1

def gen_episode(env):
    """
    A simple policy that mostly STICKs when player hand is > 18
    and mostly HITs otherwise.
    Mostly means with 80% probabilty.
    """
    episode = []
    done = False
    state = env.reset()
    while not done:
        player_hand = state[0]
        if player_hand > 18:
            action = np.random.choice([STICK, HIT], p=[0.8, 0.2])
        else:
            action = np.random.choice([STICK, HIT], p=[0.2, 0.8])
        next_state, reward, done, info = env.step(action)
        step = mcq.Step(state=state, action=action, reward=reward)
        episode.append(step)
        state = next_state
    return episode

def main():
    env = gym.make('Blackjack-v0')
    num_eps = 10
    episodes = [gen_episode(env) for _ in range(num_eps)]
    Q = mcq.learn(episodes, 2)
    for state, action_values in Q.items():
        for action, value in enumerate(action_values):
            print(f'{state}-{action}: {value}')
    env.close()


if __name__ == '__main__':
    main()
