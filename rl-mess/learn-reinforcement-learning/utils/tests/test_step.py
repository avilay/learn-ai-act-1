import numpy as np

from utils import Step


def test_ctor():
    step = Step(state=[1.0, 2.0], action=0, reward=0, next_state=[2.0, 3.0])
    assert not step.done

    step = Step()
    assert step.state is None
    assert step.action is None
    assert step.reward is None
    assert step.next_state is None
    assert not step.done

    step = Step()
    step.state = [1.0, 2.0]
    step.action = 0
    step.reward = 1
    step.next_state = [2.0, 3.0]
    step.done = True

    # The above should just work without any errors.
    assert True


def test_stripe():
    steps = []
    for _ in range(5):
        step = Step(
            state=np.random.standard_normal((3, 2)),
            action=np.random.standard_normal(2),
            reward=1.0,
            next_state=np.random.standard_normal((3, 2)),
        )
        steps.append(step)

    batch = Step.stripe(steps)

    # Check that the lengths are ok
    assert len(batch["states"]) == 5
    assert len(batch["actions"]) == 5
    assert len(batch["rewards"]) == 5
    assert len(batch["dones"]) == 5

    # Now check each element
    for i, state in enumerate(batch["states"]):
        assert np.array_equal(state, steps[i].state)

    for i, action in enumerate(batch["actions"]):
        assert np.array_equal(action, steps[i].action)

    for i, reward in enumerate(batch["rewards"]):
        assert reward == steps[i].reward

    for i, next_state in enumerate(batch["next_states"]):
        assert np.array_equal(next_state, steps[i].next_state)

    for i, done in enumerate(batch["dones"]):
        assert done == steps[i].done


def test_stripe_single():
    step = Step(state=1, action=0, reward=100, next_state=2, done=True)
    batch_of_one = Step.stripe([step])
    assert len(batch_of_one["states"]) == 1
    assert len(batch_of_one["actions"]) == 1
    assert len(batch_of_one["rewards"]) == 1
    assert len(batch_of_one["next_states"]) == 1
    assert len(batch_of_one["dones"]) == 1
    assert batch_of_one["states"][0] == 1
    assert batch_of_one["actions"][0] == 0
    assert batch_of_one["rewards"][0] == 100
    assert batch_of_one["next_states"][0] == 2
    assert batch_of_one["dones"][0] == True
