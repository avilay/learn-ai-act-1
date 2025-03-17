from typing import Callable, Dict, Tuple, Union

import numpy as np

State = Union[int, np.ndarray]
Action = Union[int, np.ndarray]
Mdp = Tuple[Dict[Tuple[State, Action, State], float], Dict[Tuple[State, Action, State], float]]
Policy = Callable[[Action, State], float]
