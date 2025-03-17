"""
Demonstrates how to get results from tasks as they get done, without having to wait for all the tasks
to get done.

$ ray submit cluster.yml simple_cluster_progress.yml
"""

from collections import Counter
import socket
import time
import ray


# ray.init()
ray.init(address="auto")


@ray.remote
def f():
    time.sleep(0.05)
    return socket.gethostbyname(socket.gethostname())


def main():
    print(
        f"""
This cluster consists of
    {len(ray.nodes())} nodes in total
    {ray.cluster_resources()["CPU"]} CPU resources in total
"""
    )
    print(f"This node's IP is {socket.gethostbyname(socket.gethostname())}")

    promises = [f.remote() for _ in range(10_000)]
    ip_addrs = []

    while promises:
        done, promises = ray.wait(promises)
        ip_addr = ray.get(done)[0]
        ip_addrs.append(ip_addr)
        if len(promises) % 250 == 0:
            print(f"{len(promises)} jobs remaining.")

    print(f"{len(ip_addrs)} tasks executed.")
    for ip_addr, num_tasks in Counter(ip_addrs).items():  # type: ignore
        print(f"    {num_tasks} tasks on {ip_addr}")


if __name__ == "__main__":
    main()
