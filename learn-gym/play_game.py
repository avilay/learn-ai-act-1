import gym
from gym.utils.play import play


def print_keys(env):
    keys_action = env.unwrapped.get_keys_to_action()
    keys = sorted(keys_action, key=keys_action.get)
    meanings = env.unwrapped.get_action_meanings()
    for action, (key, meaning) in enumerate(zip(keys, meanings)):
        chars = []
        for sub_key in key:
            sub_char = chr(sub_key)
            if sub_char == ' ':
                sub_char = 'SPC'
            chars.append(sub_char)
        print(action, chars, meaning)


def check_actions(env):
    env.reset()
    while True:
        action = int(input('Action: '))
        for frame in range(128):
            env.render()
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()


def main():
    env = gym.make('SpaceInvaders-v0')
    print_keys(env)
    # check_actions(env)
    play(env, zoom=3)

if __name__ == '__main__':
    main()
