"""VPG.

Usage:
  vpg train ENV --hparams=<hparams> --runid=<runid> [--pnet=<pnet>] [--vnet=<vnet>]
  vpg play ENV [--pnet=<pnet>] [--vnet=<vnet>]
  vpg (-h | --help)

Arguments:
  ENV   Environment name.

Options:
  -h --help             Show this screen.
  --hparams=<hparams>   Hyper parameters file in .ini format.
  --runid=<runid>       Any unique string to identify the tensorboard and checkpoint files.
  --pnet=<pnet>         Weights file of the trained policy network.
  --vnet=<vnet>         Weights file of the trained value network.
"""
from collections import deque, namedtuple
from configparser import ConfigParser
import logging
import os
import pickle

import gym
import numpy as np
import tensorflow as tf
from docopt import docopt

import utils

from .policy_net import PolicyNet
from .value_net import ValueNet


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

Hyperparams = namedtuple(
    "Hyperparams",
    [
        "mini_batch_size",
        "num_policy_iters",
        "num_value_iters_per_policy_iter",
        "policy_lr",
        "value_lr",
    ],
)


def load_hparams(filename):
    config = ConfigParser()
    config.read(filename)
    default = config["DEFAULT"]
    return Hyperparams(
        mini_batch_size=default.getint("MiniBatchSize"),
        num_policy_iters=default.getint("NumPolicyIters"),
        num_value_iters_per_policy_iter=default.getint("NumValueItersPerPolicyIter"),
        policy_lr=default.getfloat("PolicyLR"),
        value_lr=default.getfloat("ValueLR"),
    )


def play(envname, pnet_file=None, vnet_file=None):
    def policy(state):
        pass

    utils.play(env, policy)


def policy_loss_fn(returns, state_vals, probs, batch_size):
    # In this implementation advantage function is calculated as G - V(s)
    # An alternate implementation is the TD value R + γV(s') - V(s)
    advantage = returns - state_vals

    # Instead of calculating the loss as ΣA.log π(a|s) normalize the advantage value first.
    norm_advantage = (advantage - tf.math.reduce_mean(advantage)) / tf.math.reduce_std(advantage)
    return -tf.math.reduce_sum(tf.math.log(probs) * norm_advantage) / batch_size


def value_loss_fn(returns, svals):
    return tf.math.reduce_mean(tf.math.pow((returns - svals), 2))


def gen_policy(pnet):
    def policy(state):
        batch_of_one = np.expand_dims(state, axis=0)
        action_probs = pnet(batch_of_one)
        action_probs = action_probs.numpy().squeeze()
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        return action

    return policy


def log_metrics(step, avg_score, last_score, policy_loss, value_loss, pnet, vnet):
    tf.summary.scalar("Avg Score", avg_score, step)
    tf.summary.scalar("Last Score", last_score, step)
    tf.summary.scalar("Value Loss", value_loss, step)
    tf.summary.scalar("Policy Loss", policy_loss, step)
    for layer in vnet.layers:
        for j, weight in enumerate(layer.get_weights()):
            tf.summary.histogram(f"vnet-{layer.name}-{j}", weight, step)

    for layer in pnet.layers:
        for j, weight in enumerate(layer.get_weights()):
            tf.summary.histogram(f"pnet-{layer.name}-{j}", weight, step)

    print(f"Step {step}: Avg Score={avg_score:.3f}, Value Loss={value_loss:.3f}")


def train(runid, envname, hparams_file, pnet_file=None, vnet_file=None):
    hparams = load_hparams(hparams_file)
    print(f"Starting run {runid} with the following hyper parameters:")
    print(hparams)

    env = gym.make(envname)
    buf = utils.TrajectoryBuffer(env, capacity=10000)
    replinish_frac = hparams.mini_batch_size / 10000

    pnet = PolicyNet(env.action_space.n)
    if pnet_file:
        utils.load_model(pnet, env.observation_space.shape, pnet_file)
    pnet_optim = tf.keras.optimizers.Adam(learning_rate=hparams.policy_lr)

    vnet = ValueNet()
    if vnet_file:
        utils.load_model(vnet, env.observation_space.shape, vnet_file)
    vnet_optim = tf.keras.optimizers.Adam(learning_rate=hparams.value_lr)

    tbfile = f"logs/vpg-{env.spec.id}-{runid}.tb"
    print(f"Writing tensorboard metrics to {tbfile}.")
    writer = tf.summary.create_file_writer(tbfile)
    with writer.as_default():
        for i in range(hparams.num_policy_iters):
            try:
                policy = gen_policy(pnet)
                buf.replinish(policy, replinish_frac)
                batch = utils.Trajectory.stripe(buf.sample(hparams.mini_batch_size))
                with tf.GradientTape() as policy_tape:
                    all_action_probs = pnet(batch["states"])
                    action_mask = tf.one_hot(batch["actions"], env.action_space.n, dtype=tf.float32)
                    probs = tf.reduce_sum(action_mask * all_action_probs, axis=1)
                    state_vals = vnet(batch["states"])
                    policy_loss = policy_loss_fn(
                        batch["returns"], state_vals, probs, hparams.mini_batch_size
                    )
                policy_grads = policy_tape.gradient(policy_loss, pnet.trainable_variables)
                pnet_optim.apply_gradients(zip(policy_grads, pnet.trainable_variables))

                for _ in range(hparams.num_value_iters_per_policy_iter):
                    with tf.GradientTape() as value_tape:
                        state_vals = vnet(batch["states"])
                        value_loss = value_loss_fn(batch["returns"], state_vals)
                    value_grads = value_tape.gradient(value_loss, vnet.trainable_variables)
                    vnet_optim.apply_gradients(zip(value_grads, vnet.trainable_variables))

                if i % 1 == 0:
                    log_metrics(
                        i // 1, buf.avg_score, buf.latest_score, policy_loss, value_loss, pnet, vnet
                    )
            except KeyboardInterrupt:
                print("\nStopping training.")
                break

    pnet_file = f"models/vpg-{env.spec.id}-{runid}-{i}-pnet.tf"
    pnet.save_weights(pnet_file, save_format="tf")

    vnet_file = f"models/vpg-{env.spec.id}-{runid}-{i}-vnet.tf"
    vnet.save_weights(vnet_file, save_format="tf")

    print(f"Model weights saved as-\n{pnet_file}\n{vnet_file}")


def main():
    args = docopt(__doc__, version="{NAME} 1.0")
    if args["train"]:
        train(args["--runid"], args["ENV"], args["--hparams"], args["--pnet"], args["--vnet"])
    elif args["play"]:
        play(args["ENV"], args["--pnet"], args["--vnet"])


if __name__ == "__main__":
    main()
