import numpy as np
import tensorflow as tf


def play(env, policy):
    another_episode = True
    episode = 0
    while another_episode:
        print(f"\nStarting episode {episode}")
        state = env.reset()
        env.render()
        done = False
        rewards = []
        t = 0
        while not done:
            action = policy(state)
            state, reward, done, _ = env.step(action)
            t += 1
            env.render()
            rewards.append(reward)
        print(f"Episode {episode} took {t} timesteps with a total reward of {sum(rewards):.2f}.")
        ans = input("Another episode (Y/n)? ")
        another_episode = True if ans.lower().startswith("y") else False
        episode += 1

    env.close()


def load_model(net, state_dim, weights_file):
    rand_input = np.random.random(state_dim)
    rand_input = np.expand_dims(rand_input, axis=0)
    net(rand_input)
    net.load_weights(weights_file)


def log_metrics(step, avg_score, last_score, loss, net=None):
    tf.summary.scalar("Loss", loss, step)
    tf.summary.scalar("Avg Score", avg_score, step)
    tf.summary.scalar("Last Score", last_score, step)
    if net:
        for layer in net.layers:
            for j, weight in enumerate(layer.get_weights()):
                tf.summary.histogram(f"{layer.name}-{j}", weight, step)
    print(f"\rStep {step} Avg score: {avg_score:.3f}\t", end="")
