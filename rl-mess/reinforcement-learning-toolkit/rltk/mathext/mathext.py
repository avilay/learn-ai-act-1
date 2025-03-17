import math
from typing import Dict, Any


def has_converged(hsh1: Dict[Any, float], hsh2: Dict[Any, float]) -> bool:
    for k in hsh1.keys():
        v1 = hsh1[k]
        v2 = hsh2[k]
        if not math.isclose(v1, v2):
            return False
    return True
