import pytest
from rltk.simple import SimplePolicy


def test_ctor(states, actions):
    # happy day test case with probs defined for all states x actions and adding upto 1.
    s0, s1, _ = states
    a0, a1 = actions

    full_rules = [
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a0, 0.1),
        (s1, a1, 0.9)
    ]
    pi = SimplePolicy(full_rules)
    assert pi

    # some states x actions but all adding upto 1.
    partial_rules = [
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a1, 1.0)
    ]
    pi = SimplePolicy(partial_rules)
    assert pi

    # all states x actions but some states' prob not adding upto 1.
    with pytest.raises(ValueError):
        full_rules_bad = [
            (s0, a0, 0.6),
            (s0, a1, 0.3),  # s0 is not adding upto 1.
            (s1, a0, 0.1),
            (s1, a1, 0.9)
        ]
        SimplePolicy(full_rules_bad)

    # some states x actions and some not adding upto 1.
    with pytest.raises(ValueError):
        partial_rules_bad = [
            (s0, a0, 0.6),
            (s0, a1, 0.4),
            (s1, a1, 0.9)  # s1 is not adding upto 1.
        ]
        SimplePolicy(partial_rules_bad)


def test_deterministic(states, actions):
    s0, s1, _ = states
    a0, a1 = actions

    # happy case
    rules = [
        (s0, a0),
        (s1, a1)
    ]
    pi = SimplePolicy.deterministic(rules)
    assert pi

    rules = [(s0, a0)]
    pi = SimplePolicy.deterministic(rules)
    assert pi

    # adding to more than 1
    with pytest.raises(ValueError):
        rules = [
            (s0, a0),
            (s0, a1)
        ]
        pi = SimplePolicy.deterministic(rules)


def test_prob(states, actions):
    # happy day test case with probs defined for all states x actions and adding upto 1.
    s0, s1, s2 = states
    a0, a1 = actions

    full_rules = [
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a0, 0.1),
        (s1, a1, 0.9)
    ]
    pi = SimplePolicy(full_rules)
    assert 0.6 == pi.prob(a0, given=s0)
    assert 0.4 == pi.prob(a1, given=s0)
    assert 0.1 == pi.prob(a0, given=s1)
    assert 0.9 == pi.prob(a1, given=s1)
    with pytest.raises(ValueError):
        pi.prob(a0, given=s2)

    # some states x actions but all adding upto 1.
    partial_rules = [
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a1, 1.0)
    ]
    pi = SimplePolicy(partial_rules)
    assert 0.6 == pi.prob(a0, given=s0)
    assert 0.4 == pi.prob(a1, given=s0)
    assert 0.0 == pi.prob(a0, given=s1)
    assert 1.0 == pi.prob(a1, given=s1)


def test_action(states, actions):
    # happy day test case with probs defined for all states x actions and adding upto 1.
    s0, s1, s2 = states
    a0, a1 = actions

    full_rules = [
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a0, 0.1),
        (s1, a1, 0.9)
    ]
    pi = SimplePolicy(full_rules)
    for s in [s0, s1]:
        assert pi.action(s) in [a0, a1]
    with pytest.raises(ValueError):
        pi.action(s2)

    # some states x actions but all adding upto 1.
    partial_rules = [
        (s0, a0, 0.6),
        (s0, a1, 0.4),
        (s1, a1, 1.0)
    ]
    pi = SimplePolicy(partial_rules)
    assert a1 == pi.action(s1)
