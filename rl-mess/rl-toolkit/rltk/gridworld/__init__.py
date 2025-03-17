from .grid_action import GridAction
from .grid_mdp import GridMDP
from ..simple import SimpleStateValues, SimpleActionValues, SimpleEnv, SimplePolicy
from rltk import Kit


def use_gridworld_kit():
    kit = Kit.instance()
    kit.state_values = SimpleStateValues
    kit.policy = SimplePolicy
    kit.mdp = GridMDP
    kit.env = SimpleEnv
    kit.discrete_mdp = GridMDP
    kit.action_values = SimpleActionValues
