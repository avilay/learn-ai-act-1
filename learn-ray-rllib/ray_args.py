"""
Demonstrates how to pass CLI args to the script and how to have a task dependent on the output
of another task.

$ python ray_args.py --seq
2021-06-25 23:26:29,635	INFO services.py:1272 -- View the Ray dashboard at http://127.0.0.1:8266
(pid=63401) Starting task one with 10
(pid=63401) Starting task two 20
It took 2.073 seconds to calculate 40

$ python ray_args.py --no-seq
2021-06-25 23:26:43,220	INFO services.py:1272 -- View the Ray dashboard at http://127.0.0.1:8266
(pid=63499) Starting task two 20
(pid=63502) Starting task one with 10
It took 1.081 seconds to calculate [20, 40]


$ ray submit aws-cluster/snippets.yml ray_args.py -- --seq
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 06:25:29,592	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
(pid=None) Starting task one with 10
(pid=None) Starting task two 20
It took 2.419 seconds to calculate 40
Shared connection to 52.10.109.67 closed.

$ ray submit aws-cluster/snippets.yml ray_args.py -- --no-seq
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 06:25:40,094	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
(pid=None) Starting task one with 10
(pid=None) Starting task two 20
It took 1.984 seconds to calculate [20, 40]
Shared connection to 52.10.109.67 closed.

"""

import ray
import time
from datetime import datetime
import click


ray.init()
# ray.init(address="auto")


@ray.remote
def task_one(value):
    print(f"Starting task one with {value}")
    time.sleep(1)
    return value + 10


@ray.remote
def task_two(value):
    print(f"Starting task two {value}")
    time.sleep(1)
    return value * 2


@click.command()
@click.option("--seq/--no-seq", default=False)
def main(seq):
    start = datetime.now()

    if seq:
        # When passing a promise, ray will automatically resolve
        # it to return value.
        # but the tasks will execute sequentially
        promise_1 = task_one.remote(10)
        promise_2 = task_two.remote(promise_1)
        final_value = ray.get(promise_2)
    else:
        # Here the tasks are not dependant on each other
        # so they will execute in parallel
        promise_1 = task_one.remote(10)
        promise_2 = task_two.remote(20)
        final_value = ray.get([promise_1, promise_2])

    end = datetime.now()
    print(
        f"It took {(end-start).total_seconds():.3f} seconds to calculate {final_value}"
    )


if __name__ == "__main__":
    main()
