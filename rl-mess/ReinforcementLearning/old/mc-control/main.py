from common import STICK, HIT
import gym
import gym.spaces
import first_visit as simple
import first_visit_alpha_glie as glie

def Q_to_csv(Q):
    with open('./outputs/q_state_values.csv', 'wt') as f:
        print('player_hand,dealer_card,has_usable_ace,action,value', file=f)
        for state, action_values in Q.items():
            player_hand = state[0]
            dealer_card = state[1]
            has_usable_ace = state[2] == 1
            for action, action_value in enumerate(action_values):
                action_name = 'STICK' if action == STICK else 'HIT'
                print(f'{player_hand},{dealer_card},{has_usable_ace},{action_name},{action_value:.3f}', file=f)

def policy_to_csv(policy, states):
    with open('./outputs/policy.csv', 'wt') as f:
        print('player_hand,dealer_card,has_usable_ace,stick_prob,hit_prob', file=f)
        for state in states:
            player_hand = state[0]
            dealer_card = state[1]
            has_usable_ace = state[2]
            stick_prob = policy(state, STICK)
            hit_prob = policy(state, HIT)
            print(f'{player_hand},{dealer_card},{has_usable_ace},{stick_prob},{hit_prob}', file=f)

def main(is_simple):
    env = gym.make('Blackjack-v0')
    learn = simple.learn if is_simple else glie.learn
    Q, policy = learn(env, num_episodes=500000)
    Q_to_csv(Q)
    policy_to_csv(policy, Q.keys())


if __name__ == '__main__':
    main(is_simple=False)
