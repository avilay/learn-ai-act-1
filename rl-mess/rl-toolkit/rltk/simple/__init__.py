# flake8: noqa
from .simple_action_values import SimpleActionValues
from .simple_state_values import SimpleStateValues
from .simple_policy import SimplePolicy
from .simple_mdp import SimpleMDP
from .simple_env import SimpleEnv
from rltk import Kit


def use_simpleworld_kit():
    kit = Kit.instance()
    kit.state_values = SimpleStateValues
    kit.policy = SimplePolicy
    kit.mdp = SimpleMDP
    kit.env = SimpleEnv
    kit.discrete_mdp = SimpleMDP
    kit.action_values = SimpleActionValues
