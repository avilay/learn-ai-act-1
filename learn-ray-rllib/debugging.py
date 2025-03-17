"""
Doesn't really work.
"""

import ray

ray.init(address="auto")


@ray.remote
def f(x):
    ray.util.pdb.set_trace()
    return x * x


promises = [f.remote(i) for i in range(4)]
print(ray.get(promises))
