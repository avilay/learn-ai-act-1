# flake8: noqa E402

import logging
import uuid

import numpy as np
import pytorch_lightning as pl
import torch as t

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("reagent").setLevel(logging.ERROR)

import warnings

from reagent.core.parameters import RLParameters
from reagent.gym.agents.agent import Agent
from reagent.gym.datasets.replay_buffer_dataset import ReplayBufferDataset
from reagent.gym.envs import Gym
from reagent.gym.policies.random_policies import make_random_policy_for_env
from reagent.gym.runners.gymrunner import evaluate_for_n_episodes
from reagent.gym.utils import build_normalizer, fill_replay_buffer
from reagent.model_managers.discrete.discrete_dqn import DiscreteDQN
from reagent.net_builder.discrete_dqn.fully_connected import FullyConnected
from reagent.net_builder.unions import DiscreteDQNNetBuilder__Union
from reagent.optimizer.uninferrable_optimizers import Adam
from reagent.optimizer.union import Optimizer__Union
from reagent.replay_memory.circular_replay_buffer import ReplayBuffer
from reagent.training import DQNTrainerParameters


def build_dqn():
    rl_params = RLParameters(
        gamma=0.99,
        epsilon=0.1,
        target_update_rate=0.2,
        maxq_learning=True,
        reward_boost=None,
        temperature=1.0,
        softmax_policy=False,
        use_seq_num_diff_as_time_diff=False,
        q_network_loss="mse",
        set_missing_value_to_zero=False,
        tensorboard_logging_freq=0,
        predictor_atol_check=0.0,
        predictor_rtol_check=5e-05,
        time_diff_unit_length=1.0,
        multi_steps=None,
        ratio_different_predictions_tolerance=0.0,
    )

    adam = Adam(
        lr_schedulers=[],
        lr=0.05,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0,
        amsgrad=False,
    )
    optim = Optimizer__Union(Adam=adam)

    trainer_params = DQNTrainerParameters(
        actions=["0", "1"],
        rl=rl_params,
        double_q_learning=True,
        bcq=None,
        minibatch_size=1024,
        minibatches_per_step=1,
        optimizer=optim,
    )

    net_builder = DiscreteDQNNetBuilder__Union(
        Dueling=None,
        FullyConnected=FullyConnected(
            sizes=[128, 64],
            activations=["leaky_relu", "leaky_relu"],
            dropout_ratio=0.0,
            use_batch_norm=False,
        ),
        FullyConnectedWithEmbedding=None,
    )
    cpe_net_builder = DiscreteDQNNetBuilder__Union(
        Dueling=None,
        FullyConnected=FullyConnected(
            sizes=[256, 128],
            activations=["relu", "relu"],
            dropout_ratio=0.0,
            use_batch_norm=False,
        ),
        FullyConnectedWithEmbedding=None,
    )

    dqn = DiscreteDQN(
        trainer_param=trainer_params,
        net_builder=net_builder,
        cpe_net_builder=cpe_net_builder,
    )
    return dqn


def build_replay_buffer(env, replay_buffer):

    train_after_ts = 20000

    random_policy = make_random_policy_for_env(env)
    agent = Agent.create_for_env(env, policy=random_policy)

    fill_replay_buffer(
        env=env, replay_buffer=replay_buffer, desired_size=train_after_ts, agent=agent
    )


def train(env):
    train_every_ts = 1
    replay_memory_size = 100000
    minibatch_size = 512
    num_train_episodes = 10

    replay_buffer = ReplayBuffer(
        replay_capacity=replay_memory_size, batch_size=minibatch_size
    )
    build_replay_buffer(env, replay_buffer)

    dqn = build_dqn()
    normalization = build_normalizer(env)
    trainer = dqn.build_trainer(use_gpu=False, normalization_data_map=normalization)
    training_policy = dqn.create_policy(trainer, serving=False)
    agent = Agent.create_for_env(env, policy=training_policy)
    dataset = ReplayBufferDataset.create_for_trainer(
        trainer,
        env,
        agent,
        replay_buffer,
        batch_size=minibatch_size,
        training_frequency=train_every_ts,
        num_episodes=num_train_episodes,
        max_steps=200,
    )
    data_loader = t.utils.data.DataLoader(dataset, collate_fn=lambda batch: batch[0])
    pl_trainer = pl.Trainer(
        max_epochs=1,
        deterministic=True,
        default_root_dir=f"lightning_log_{str(uuid.uuid4())}",
        progress_bar_refresh_rate=0,
    )
    pl_trainer.fit(trainer, data_loader)

    return dqn, trainer


def main():
    num_eval_episodes = 10

    env = Gym(env_name="CartPole-v0")

    eval_rewards = evaluate_for_n_episodes(
        n=num_eval_episodes,
        env=env,
        agent=Agent.create_for_env(env, policy=make_random_policy_for_env(env)),
        max_steps=env.max_steps,
        num_processes=1,
    ).squeeze(1)
    print(f"Average reward before training: {np.mean(eval_rewards):.3f}")

    dqn, trainer = train(env)

    serving_policy = dqn.create_policy(
        trainer, serving=True, normalization_data_map=build_normalizer(env)
    )
    agent = Agent.create_for_env_with_serving_policy(env, serving_policy)
    eval_rewards = evaluate_for_n_episodes(
        n=num_eval_episodes,
        env=env,
        agent=agent,
        max_steps=env.max_steps,
        num_processes=1,
    ).squeeze(1)

    print(f"Average reward after training: {np.mean(eval_rewards):.3f}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    main()
