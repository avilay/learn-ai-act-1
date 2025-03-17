"""
Demonstrates starting a lot of tasks and having them queue up on various workers.

If the workers had already been started, then the tasks will be more-or-less uniformly distributed. Otherwise
the first worker and maybe the head node will end up running a lot of the tasks.

$ ray submit cluster.yml simple_cluster.py
"""

from collections import Counter
import socket
import time
import ray

# This will use workers if they exist or scale out and start new workers if they dont.
ray.init(address="auto")

# This will run this only on the head node, it will not use any workers.
# I think this is for a "local" cluster.
# ray.init()

print(
    f"""
This cluster consists of
    {len(ray.nodes())} nodes in total
    {ray.cluster_resources()["CPU"]} CPU resources in total
"""
)
print(f"This node's IP is {socket.gethostbyname(socket.gethostname())}")


@ray.remote
def f():
    time.sleep(0.01)
    return socket.gethostbyname(socket.gethostname())


promises = [f.remote() for _ in range(10_000)]
ip_addrs = ray.get(promises)

print("Tasks executed")
for ip_addr, num_tasks in Counter(ip_addrs).items():  # type: ignore
    print(f"    {num_tasks} tasks on {ip_addr}")
