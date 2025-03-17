from typing import Optional, Callable, Set

from haikunator import Haikunator
from mlmetrics.metric import Metric

from .core.abstract.action_values import ActionValues
from .core.abstract.discrete_mdp import DiscreteMDP
from .core.abstract.env import Env
from .core.abstract.mdp import MDP
from .core.abstract.policy import Policy
from .core.abstract.state_values import StateValues


class Kit:
    _singleton = None

    def __init__(self):
        self._name = Haikunator().haikunate()

        self._action_values_ctor: Optional[Callable[[], ActionValues]] = None
        self._discrete_mdp_ctor: Optional[Callable[[], DiscreteMDP]] = None
        self._env_ctor: Optional[Callable[[], Env]] = None
        self._mdp_ctor: Optional[Callable[[], MDP]] = None
        self._policy_ctor: Optional[Callable[[], Policy]] = None
        self._state_values_ctor: Optional[Callable[[], StateValues]] = None
        self._metric_ctor: Optional[Callable[[str, Set], Metric]] = None

    @classmethod
    def instance(cls):
        if not cls._singleton:
            cls._singleton = cls()
        return cls._singleton

    def reset(self):
        self._action_values_ctor = None
        self._discrete_mdp_ctor = None
        self._env_ctor = None
        self._mdp_ctor = None
        self._policy_ctor = None
        self._state_values_ctor = None
        self._metric_ctor = None

    @property
    def name(self) -> str:
        return self._name

    # Metric
    @property
    def metric(self):
        raise RuntimeError('metric is a write-only property')

    @metric.setter
    def metric(self, ctor: Callable[[str, Set], Metric]):
        if self._metric_ctor:
            raise RuntimeError('metric has alredy been set for this process')
        self._metric_ctor = ctor

    def new_metric(self, name: str, labels: Set) -> Metric:
        if self._metric_ctor is None:
            raise RuntimeError('metric has not been set')
        return self._metric_ctor(name=name, labels=labels)

    # ActionValues
    @property
    def action_values(self):
        raise RuntimeError('action_values is a write-only property')

    @action_values.setter
    def action_values(self, ctor: Callable[[], ActionValues]):
        if self._action_values_ctor:
            raise RuntimeError('action_values has already been set once for this process')
        self._action_values_ctor = ctor

    def new_action_values(self) -> ActionValues:
        if self._action_values_ctor is None:
            raise RuntimeError('action_values have not been set')
        return self._action_values_ctor()

    # DiscreteMDP
    @property
    def discrete_mdp(self):
        raise RuntimeError('discrete_mdp is a write-only property')

    @discrete_mdp.setter
    def discrete_mdp(self, ctor: Callable[[], DiscreteMDP]):
        if self._discrete_mdp_ctor:
            raise RuntimeError('discrete_mdp has already been set once for this process')
        self._discrete_mdp_ctor = ctor

    def new_discrete_mdp(self) -> DiscreteMDP:
        if self._discrete_mdp_ctor is None:
            raise RuntimeError('discrete_mdp has not been set')
        return self._discrete_mdp_ctor()

    # Env
    @property
    def env(self):
        raise RuntimeError('env is a write-only property')

    @env.setter
    def env(self, ctor: Callable[[], Env]):
        if self._env_ctor:
            raise RuntimeError('env has already been set once for this process')
        self._env_ctor = ctor

    def new_env(self) -> Env:
        if self._env_ctor is None:
            raise RuntimeError('Env has not been set')
        return self._env_ctor()

    # MDP
    @property
    def mdp(self):
        raise RuntimeError('mdp is a write-only property')

    @mdp.setter
    def mdp(self, ctor: Callable[[], MDP]):
        if self._mdp_ctor:
            raise RuntimeError('mdp has already been set once for this process')
        self._mdp_ctor = ctor

    def new_mdp(self) -> MDP:
        if self._mdp_ctor is None:
            raise RuntimeError('MDP has not been set')
        return self._mdp_ctor()

    # Policy
    @property
    def policy(self):
        raise RuntimeError('policy is a write-only property')

    @policy.setter
    def policy(self, ctor: Callable[[], Policy]):
        if self._policy_ctor:
            raise RuntimeError('policy has already been set once for this process')
        self._policy_ctor = ctor

    def new_policy(self) -> Policy:
        if self._policy_ctor is None:
            raise RuntimeError('Policy has not been set')
        return self._policy_ctor()

    # StateValues
    @property
    def state_values(self):
        raise RuntimeError('state_values is a write-only property')

    @state_values.setter
    def state_values(self, ctor: Callable[[], StateValues]):
        if self._state_values_ctor:
            raise RuntimeError('state_values has already been set once for this process')
        self._state_values_ctor = ctor

    def new_state_values(self) -> StateValues:
        if self._state_values_ctor is None:
            raise RuntimeError('StateValues have not been set')
        return self._state_values_ctor()

    def print_contents(self):
        print(f'action_values={self.action_values.__name__}')
        print(f'discrete_mdp={self.discrete_mdp.__name__}')
        print(f'env={self.env.__name__}')
        print(f'mdp={self.mdp.__name__}')
        print(f'policy={self.policy.__name__}')
        print(f'state_values={self.state_values.__name__}')
