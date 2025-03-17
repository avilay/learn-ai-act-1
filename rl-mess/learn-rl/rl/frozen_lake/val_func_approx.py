import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import os.path as path
import sys
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
from haikunator import Haikunator
from tensorflow.keras.layers import Dense, Input

from .val_planning import build_best_policy
from .hyperparams import Hyperparams
import rl.common.replay_buffer as replay_buffer


tf.get_logger().setLevel("ERROR")
np.set_printoptions(precision=3)


def gen_mc_dataset(env, π, num_steps):
    buf = replay_buffer.build(env, π, num_steps)
    x = []
    y = []
    for ep in buf:
        ep.calc_returns(gamma=1.0)
        for step, g in ep:
            x.append(step.state)
            y.append(g)
    x = tf.one_hot(np.array(x), depth=env.observation_space.n)
    y = np.array(y).reshape(-1, 1)
    return tf.data.Dataset.from_tensor_slices((x, y))


def build_model(n_states):
    inputs = Input(shape=(n_states,))
    output = Dense(1)(inputs)
    return tf.keras.Model(inputs, output)


def print_state_stats(ds):
    all_returns = defaultdict(list)
    for x, y in ds:
        state = tf.where(tf.equal(x, 1))[0, 0].numpy()
        return_ = y[0].numpy()
        all_returns[state].append(return_)

    stats = {}
    for state, returns in all_returns.items():
        μ = np.mean(returns)
        σ = np.std(returns)
        r = μ / σ if σ != 0 else 0.0
        stats[state] = (len(returns), μ, σ, r)

    print("State Stats -")
    for state in sorted(stats.keys()):
        n, μ, σ, r = stats[state]
        print(f"\t{state}: n={n} μ={μ:.3f} σ={σ:.3f} μ/σ={r:.3f}")
    print("\n")


def test(run_id):
    states = tf.one_hot(np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14]), depth=16)
    exp_svals = np.array(
        [
            0.0167572206,
            0.0118173959,
            0.0275739230,
            0.0118173959,
            0.0272827840,
            0.0682782830,
            0.0636598274,
            0.184916637,
            0.227594273,
            0.325134678,
            0.573730930,
        ]
    )
    test_ds = tf.data.Dataset.from_tensor_slices((states, exp_svals))
    model_file = path.expanduser(f"~/mldata/tblogs/frozen-lake/{run_id}/model.h5")
    model = tf.keras.models.load_model(model_file)
    print("\n\nTest Metrics -")
    loss, rmse = model.evaluate(test_ds.batch(len(exp_svals)))
    for x, y in test_ds:
        y_hat = model(tf.expand_dims(x, axis=0))
        y = y.numpy()
        y_hat = y_hat[0, 0].numpy()
        x = tf.where(tf.equal(x, 1))[0, 0]
        print(f"[{x}] exp_sval={y:.3f} pred_sval={y_hat:.3f}")
    return loss, rmse


def main():
    if len(sys.argv) > 1:
        hparams = Hyperparams.load(sys.argv[1])
    else:
        hparams = Hyperparams(
            batch_size=32,
            epochs=2,
            lr=0.001,
            num_train_steps=100,
            num_val_steps=10,
            activation="tanh",
        )
    run_id = Haikunator().haikunate()
    print(f"Starting run with {run_id} and {hparams}")

    fl = gym.make("FrozenLake-v0")
    policy = build_best_policy(fl)
    train_ds = gen_mc_dataset(fl, policy, hparams.num_train_steps)
    val_ds = gen_mc_dataset(fl, policy, hparams.num_val_steps)
    print_state_stats(train_ds)

    model = build_model(fl.observation_space.n)
    tblog = path.expanduser(path.join("~/mldata/tblogs/frozen-lake/", run_id))
    tb = tf.keras.callbacks.TensorBoard(tblog, histogram_freq=0, update_freq="epoch")
    optim = tf.keras.optimizers.Adam(learning_rate=hparams.lr)
    rmse = tf.metrics.RootMeanSquaredError("rmse")
    model.compile(optimizer=optim, loss="mse", metrics=[rmse])
    train_ds = train_ds.shuffle(hparams.num_train_steps).batch(hparams.batch_size)
    val_ds = val_ds.batch(hparams.num_val_steps)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=hparams.epochs, callbacks=[tb], verbose=2
    )

    model_ckpt = path.join(tblog, "model.h5")
    model.save(model_ckpt)

    writer = tf.summary.create_file_writer(tblog)
    with writer.as_default():
        tf.summary.text("hparams", str(hparams), step=1)

    test_loss, test_rmse = test(run_id)
    best_val_loss = np.min(history.history["val_loss"])
    corr_val_rmse = history.history["val_rmse"][np.argmin(history.history["val_loss"])]
    exp_file = "frozen-lake-exps.csv"
    if not path.exists(exp_file):
        with open(exp_file, "wt") as f:
            print(
                "name," + Hyperparams.csv_header() + ",val_loss,test_loss,val_rmse,test_rmse",
                file=f,
            )
    with open(exp_file, "at") as f:
        print(
            f"{run_id},{hparams.csv_repr()},{best_val_loss},{test_loss},{corr_val_rmse},{test_rmse}",
            file=f,
        )


if __name__ == "__main__":
    main()
