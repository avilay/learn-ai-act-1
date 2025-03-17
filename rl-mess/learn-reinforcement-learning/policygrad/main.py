"""PolicyGrad.

Usage:
  pg train ENV --hparams=<hparams> --runid=<runid> [--pnet=<pnet>]
  pg play ENV [--pnet=<pnet>]
  pg (-h | --help)

Arguments:
  ENV   Environment name.

Options:
  -h --help             Show this screen.
  --hparams=<hparams>   Hyper parameters file in .ini format.
  --runid=<runid>       Any unique string to identify the tensorboard and checkpoint files.
  --pnet=<pnet>         Weights file of the trained policy network.
"""
import sys
from collections import deque, namedtuple
from configparser import ConfigParser
from datetime import datetime
from pprint import pprint

import gym
import numpy as np
import tensorflow as tf
from docopt import docopt

from utils import log_metrics, play

from .agent import Agent
from .policy import Policy

Hyperparams = namedtuple("Hyperparams", ["mini_batch_size", "num_iters", "learning_rate"])


def load_hparams(filename):
    config = ConfigParser()
    config.read(filename)
    default = config["DEFAULT"]
    return Hyperparams(
        mini_batch_size=default.getint("MiniBatchSize"),
        num_iters=default.getint("NumIters"),
        learning_rate=default.getfloat("LearningRate"),
    )


@tf.function
def loss_fn(scores, probs):
    return -tf.reduce_sum(scores * tf.math.log(probs))


def play(envname, pnet_file=None):
    env = gym.make(envname)
    pnet = Policy(env.action_space.n)
    if pnet_file:
        rand_input = np.random.random(env.observation_space.shape).astype(np.float32)
        rand_input = np.expand_dims(rand_input, axis=0)
        pnet(rand_input)
        pnet.load_weights(pnet_file)
        print(f"Loaded weights from {pnet_file}.")

    def policy(state):
        state = np.expand_dims(state, axis=0)
        action_probs = pnet(state).numpy().squeeze()
        return np.argmax(action_probs)

    play(env, policy)


def train(runid, envname, hparams_file, pnet_file=None):
    hparams = load_hparams(hparams_file)
    print(f"Starting run {runid} with the following hyper parameters:")
    print(hparams)

    env = gym.make(envname)

    policy = Policy(env.action_space.n)
    if pnet_file:
        rand_input = np.random.random(env.observation_space.shape).astype(np.float32)
        rand_input = np.expand_dims(rand_input, axis=0)
        policy(rand_input)
        policy.load_weights(pnet_file)

    agent = Agent(env, hparams.mini_batch_size)
    optim = tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate)
    scores_window = deque(maxlen=20)

    tbfile = f"logs/pg-{env.spec.id}-{runid}.tb"
    print(f"Writing tensorboard metrics to {tbfile}.")
    writer = tf.summary.create_file_writer(tbfile)

    with writer.as_default():
        for i in range(hparams.num_iters):
            try:
                batch, scores = agent.gen_batch(policy)
                scores_window.extend(scores)
                with tf.GradientTape() as tape:
                    # size is m x |A|
                    all_action_probs = policy(batch.states)

                    # need it to be m
                    action_mask = tf.one_hot(batch.actions, env.action_space.n, dtype=tf.float32)
                    probs = tf.reduce_sum(action_mask * all_action_probs, axis=1)
                    # Now probs is π(a|s)

                    loss = loss_fn(batch.returns, probs)
                grads = tape.gradient(loss, policy.trainable_variables)
                optim.apply_gradients(zip(grads, policy.trainable_variables))

                if i % 1 == 0:
                    log_metrics(i // 1, scores_window, loss, policy)

            except KeyboardInterrupt:
                print("\nStopping training.")
                break

    filename = f"logs/pg-{env.spec.id}-{runid}-{i}-pnet.tf"
    policy.save_weights(filename, save_format="tf")
    print(f"Model weights saved as {filename}")


def main():
    args = docopt(__doc__, version="PolicyGrad 1.0")
    if args["train"]:
        train(args["--runid"], args["ENV"], args["--hparams"], args["--pnet"])
    elif args["play"]:
        play(args["ENV"], args["--pnet"])


if __name__ == "__main__":
    main()
