"""DQN.

Usage:
  dqn train ENV --hparams=<hparams> --runid=<runid> [--qnet=<qnet>]
  dqn play ENV [--qnet=<qnet>]
  dqn (-h | --help)

Arguments:
  ENV   Environment name.

Options:
  -h --help                 Show this screen.
  --hparams=<hparams>       Hyper parameters file in .ini format.
  --runid=<runid>           Any unique string to identify tensorboard and checkpoint files.
  --qnet=<qnet>             Weights file of the trained Q-network.
"""

import logging
import signal
import sys
from collections import namedtuple
from configparser import ConfigParser
from pprint import pprint

import gym
import numpy as np
import shortuuid
import tensorflow as tf
from docopt import docopt

from utils import play

from .dqn import DQN
from .replay_buffer import ReplayBuffer

Hyperparams = namedtuple(
    "Hyperparams",
    [
        "capacity",
        "eps_0",
        "eps_end",
        "num_iters",
        "mini_batch_size",
        "update_freq",
        "eps_decay",
        "replinish_pct",
        "gamma",
        "tau",
        "learning_rate",
    ],
)


def load_hparams(filename):
    # config = ConfigParser(filename)["DEFAULT"]
    config = ConfigParser()
    config.read(filename)
    default = config["DEFAULT"]
    return Hyperparams(
        capacity=default.getint("Capacity"),
        eps_0=default.getfloat("Eps0"),
        eps_end=default.getfloat("EpsEnd"),
        num_iters=default.getint("NumIters"),
        mini_batch_size=default.getint("MiniBatchSize"),
        update_freq=default.getint("UpdateFreq"),
        eps_decay=default.getfloat("EpsDecay"),
        replinish_pct=default.getfloat("ReplinishPct"),
        gamma=default.getfloat("Gamma"),
        tau=default.getfloat("Tau"),
        learning_rate=default.getfloat("LearningRate"),
    )


def play(envname, qnet_file=None):
    env = gym.make(envname)
    Q = DQN(env.action_space.n)
    rand_input = np.random.random(env.observation_space.shape)
    rand_input = np.expand_dims(rand_input, axis=0)
    Q(rand_input)

    if qnet_file:
        print(f"Loading weights from {qnet_file}")
        Q.load_weights(qnet_file)

    def policy(state):
        state = np.expand_dims(state, axis=0)
        return np.argmax(Q(state))

    play(env, policy)


def train(runid, envname, hparams_file, qnet_file=None):
    hparams = load_hparams(hparams_file)
    print(f"Starting run {runid} with the following hyper parameters:")
    pprint(hparams)

    env = gym.make(envname)
    Q = DQN(env.action_space.n)
    Q_target = DQN(env.action_space.n)

    # Run the network on a random input to generate the weights
    rand_input = np.random.random(env.observation_space.shape)
    rand_input = np.expand_dims(rand_input, axis=0)
    Q(rand_input)
    Q_target(rand_input)

    if qnet_file:
        print(f"Loading weights from {qnet_file}")
        Q.load_weights(qnet_file)

    # Q_target should start with the same weights as Q
    Q_target.set_weights(Q.get_weights())

    # Create the replay buffer and fill it before starting training
    eps = hparams.eps_0
    buf = ReplayBuffer(env, hparams.capacity)
    buf.replinish(hparams.mini_batch_size / hparams.capacity, Q, eps)

    loss_fn = tf.keras.losses.MeanSquaredError()
    optim = tf.keras.optimizers.Adam(learning_rate=hparams.learning_rate)

    tbfile = f"logs/{env.spec.id}-{runid}.tb"
    print(f"Writing tensorboard summary in {tbfile}.")
    writer = tf.summary.create_file_writer(tbfile)
    with writer.as_default():
        for i in range(hparams.num_iters):
            try:
                # print(f"\rStarting iteration {i}.", end="")
                # Update Q
                batch = buf.sample(hparams.mini_batch_size)
                with tf.GradientTape() as tape:
                    # This gives the qvals for all actions for each state in the batch
                    # i.e., pred_vals is of size batch_size x action_space
                    # We only need to select those actions that are in the batch
                    pred_qvals = Q(batch.states)
                    one_hot_mask = (
                        tf.one_hot(batch.actions, env.action_space.n)
                        .numpy()
                        .squeeze()
                        .astype(np.bool)
                    )
                    pred_qvals = pred_qvals[one_hot_mask]

                    # This gives only one qval for each state in the batch. The sizes of pred and target
                    # should match.
                    max_qval_next = np.max(Q_target(batch.next_states), axis=1)
                    target_qvals = batch.rewards + (
                        hparams.gamma * max_qval_next * (1 - batch.dones)
                    )
                    loss = loss_fn(target_qvals, pred_qvals)
                grads = tape.gradient(loss, Q.trainable_variables)
                optim.apply_gradients(zip(grads, Q.trainable_variables))

                # Soft update Q_target
                w_target = np.array(Q_target.get_weights())
                w = np.array(Q.get_weights())
                w_target = (1 - hparams.tau) * w_target + hparams.tau * w
                Q_target.set_weights(w_target)

                # Replinish buffer if needed
                if i % hparams.update_freq == 0:
                    eps = max(eps * hparams.eps_decay, hparams.eps_end)
                    buf.replinish(hparams.replinish_pct, Q, eps)

                # Write out the summary every 100 iters
                if i % 50 == 0:
                    step = i // 50
                    tf.summary.scalar("Loss", loss, step)
                    tf.summary.scalar("EPS", eps, step)
                    tf.summary.scalar("Avg Score", np.mean(buf.avg_score), step)
                    tf.summary.scalar("Last Score", buf.last_score, step)
                    # for layer in Q.layers:
                    #     for j, weight in enumerate(layer.get_weights()):
                    #         tf.summary.histogram(f"{layer.name}-{j}", weight, step)

                    print(f"\rIter {i} Avg score: {buf.avg_score:.2f}\t", end="")
            except KeyboardInterrupt:
                print("\nStopping training.")
                break

    filename = f"logs/dqn-{env.spec.id}-{runid}-{i}-qnet.tf"
    Q.save_weights(filename, save_format="tf")
    print(f"Model weights saved as {filename}")


def main():
    logging.basicConfig(level=logging.ERROR)
    args = docopt(__doc__, version="DQN 1.0")
    if args["train"]:
        train(args["--runid"], args["ENV"], args["--hparams"], args["--qnet"])
    elif args["play"]:
        play(args["ENV"], args["--qnet"])


if __name__ == "__main__":
    main()
