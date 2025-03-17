import gym
import numpy as np

from rl.common.planning import calc_state_vals

from rl.frozen_lake.val_planning import build_best_policy
from . import simple


def test_calc_state_vals():
    env = simple.SimpleEnv()
    svals = calc_state_vals(env, simple.policy)
    exp_svals = np.array([8.03114032, 11.17119132, 8.92357587])
    assert np.allclose(exp_svals, svals)


def test_calc_state_vals_fl():
    frozen_lake = gym.make("FrozenLake-v0")
    policy = build_best_policy(frozen_lake)
    exp_svals = np.array(
        [
            0.0167572206,
            0.0118173959,
            0.0275739230,
            0.0118173959,
            0.0272827840,
            0.0,
            0.0682782830,
            0.0,
            0.0636598274,
            0.184916637,
            0.227594273,
            0.0,
            0.0,
            0.325134678,
            0.573730930,
            0.0000000969,
        ]
    )
    svals = calc_state_vals(frozen_lake, policy)
    assert np.allclose(exp_svals, svals)
