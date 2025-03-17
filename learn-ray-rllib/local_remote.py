"""
Demonstrates how to avoid having to make code changes when going from local to remote.
Another way is to just start a local server with start_local.py.
"""

import time
import socket

import ray


def init_ray():
    is_local = socket.gethostname() == "avilay-mbp"
    ray.init() if is_local else ray.init(address="auto")


@ray.remote
def task():
    print("Doing task...")
    time.sleep(3)
    return 42


def main():
    print(socket.gethostname())
    init_ray()
    promise = task.remote()
    # ray.get([promise])
    val = ray.get(promise)
    print(f"Got value {val}")


if __name__ == "__main__":
    main()
