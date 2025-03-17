from collections import Counter
import socket
import time
import ray
import logging
from snippets.log_config import configure_logger


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


@ray.remote
def f():
    configure_logger()
    logging.info("This is info log")
    logging.debug("This is debug log")
    logging.info("Sleeping for 0.001 seconds")
    time.sleep(0.001)
    logging.info("Awake now")
    return socket.gethostbyname(socket.gethostname())


promises = [f.remote() for _ in range(10_000)]
ip_addrs = ray.get(promises)

print("Tasks executed")
for ip_addr, num_tasks in Counter(ip_addrs).items():  # type: ignore
    print(f"    {num_tasks} tasks on {ip_addr}")
