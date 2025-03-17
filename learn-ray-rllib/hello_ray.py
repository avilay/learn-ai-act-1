"""
Demos a simple ray task that returns a promise.

To run the program locally -

$ python hello_ray.py
2021-06-25 23:56:14,117	INFO worker.py:726 -- Connecting to existing Ray cluster at address: 192.168.1.3:6379
(pid=69272) Starting task...
Got value 42

To run the program remotely -
$ ray submit aws-cluster/snippets.yml hello_ray.py
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 06:56:42,461	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
(pid=29912) Starting task...
Got value 42
Shared connection to 52.10.109.67 closed.

"""

import ray
import time


ray.init(address="auto")


@ray.remote
def task():
    print("Starting task...")
    time.sleep(3)
    return 42


def main():
    promise = task.remote()
    val = ray.get(promise)
    print(f"Got value {val}")


if __name__ == "__main__":
    main()
