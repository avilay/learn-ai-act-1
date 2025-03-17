import numpy as np

from utils import Step, Trajectory


def test_add_step():
    t = Trajectory()
    t.add_step(Step(state=[1.0, 2.0], action=0, reward=10))
    assert t.total_score == 10
    t.add_step(Step(state=[3.0, 4.0], action=1, reward=-8))
    assert t.total_score == 2
    assert len(t._steps) == 2


def test_len():
    t = Trajectory()
    t.add_step(Step(state=[1.0, 2.0], action=0, reward=10))
    t.add_step(Step(state=[3.0, 4.0], action=1, reward=-8))
    assert len(t) == 2


def test_getitem():
    t = Trajectory()

    s0 = Step(state=[1.0, 2.0], action=0, reward=10)
    t.add_step(s0)

    s1 = Step(state=[3.0, 4.0], action=1, reward=-8)
    t.add_step(s1)

    assert t[0] == (s0, None)
    assert t[1] == (s1, None)

    t.calculate_returns()
    assert t[0][1] is not None
    assert t[1][1] is not None


def test_calc_returns():
    t = Trajectory(discount_factor=1.0)
    t.add_step(Step(state=[1.0, 2.0], action=0, reward=10))
    t.add_step(Step(state=[3.0, 4.0], action=1, reward=-8))
    returns = t.calculate_returns()
    assert returns[0] == 2
    assert returns[1] == -8

    t = Trajectory(discount_factor=0)
    t.add_step(Step(state=[1.0, 2.0], action=0, reward=10))
    t.add_step(Step(state=[3.0, 4.0], action=1, reward=-8))
    returns = t.calculate_returns()
    assert returns[0] == 10
    assert returns[1] == -8

    t = Trajectory()
    t.add_step(Step(state=[1.0, 2.0], action=0, reward=10))
    t.add_step(Step(state=[3.0, 4.0], action=1, reward=-8))
    t.add_step(Step(state=[5.0, 6.0], action=0, reward=5))
    returns = t.calculate_returns()
    assert returns[0] == 10 + 0.99 * (-8) + 0.99 * 0.99 * 5
    assert returns[1] == (-8) + 0.99 * 5
    assert returns[2] == 5


def test_stripe():
    t0 = Trajectory(discount_factor=1)
    s00 = Step(state=1, action=1, reward=10)
    s01 = Step(state=2, action=0, reward=9)
    s02 = Step(state=3, action=1, reward=8)
    t0.add_step(s00)
    t0.add_step(s01)
    t0.add_step(s02)
    t0.calculate_returns()

    t1 = Trajectory(discount_factor=1)
    s10 = Step(state=4, action=-1, reward=-10)
    s11 = Step(state=5, action=1, reward=10)
    t1.add_step(s10)
    t1.add_step(s11)
    t1.calculate_returns()

    all_steps = t0._steps + t1._steps
    all_returns = t0._returns + t1._returns

    batch = Trajectory.stripe([t0, t1])

    assert len(batch["states"]) == len(all_steps)
    for i, state in enumerate(batch["states"]):
        assert state == all_steps[i].state

    assert len(batch["actions"]) == len(all_steps)
    for i, action in enumerate(batch["actions"]):
        assert action == all_steps[i].action

    assert len(batch["rewards"]) == len(all_steps)
    for i, reward in enumerate(batch["rewards"]):
        assert reward == all_steps[i].reward

    assert len(batch["next_states"]) == len(all_steps)
    for i, next_state in enumerate(batch["next_states"]):
        assert next_state == all_steps[i].next_state

    assert len(batch["dones"]) == len(all_steps)
    for i, done in enumerate(batch["dones"]):
        assert done == all_steps[i].done

    assert len(batch["returns"]) == len(all_returns)
    for i, return_ in enumerate(batch["returns"]):
        assert return_ == all_returns[i]


def test_stripe_single():
    t0 = Trajectory(discount_factor=1)
    s00 = Step(state=1, action=1, reward=10)
    s01 = Step(state=2, action=0, reward=9)
    s02 = Step(state=3, action=1, reward=8)
    t0.add_step(s00)
    t0.add_step(s01)
    t0.add_step(s02)
    t0.calculate_returns()

    batch_of_one = Trajectory.stripe([t0])

    assert len(batch_of_one["states"]) == len(t0)
    exp = np.array([step.state for step, _ in t0])
    act = batch_of_one["states"]
    assert np.array_equal(exp, act)

    assert len(batch_of_one["actions"]) == len(t0)
    exp = np.array([step.action for step, _ in t0])
    act = batch_of_one["actions"]
    assert np.array_equal(exp, act)

    assert len(batch_of_one["rewards"]) == len(t0)
    exp = np.array([step.reward for step, _ in t0])
    act = batch_of_one["rewards"]
    assert np.array_equal(exp, act)

    assert len(batch_of_one["next_states"]) == len(t0)
    exp = np.array([step.next_state for step, _ in t0])
    act = batch_of_one["next_states"]
    assert np.array_equal(exp, act)

    assert len(batch_of_one["dones"]) == len(t0)
    exp = np.array([step.done for step, _ in t0])
    act = batch_of_one["dones"]
    assert np.array_equal(exp, act)

    assert len(batch_of_one["returns"]) == len(t0)
    exp = np.array([return_ for _, return_ in t0])
    act = batch_of_one["returns"]
    assert np.array_equal(exp, act)
