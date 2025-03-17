"""
Demonstrates the async nature of ray tasks. Each of the 10 tasks takes 1 second to run, when run without ray,
these are run sequentially and the entire program takes ~10 seconds to run. When run with ray,
the program will run in 1 second. If I see longer times on the remote cluster that is probably because the
worker instances are not fully up yet.

To run this program locally ensure that `ray.init()` is active and `ray.init(address="true")` is commented out.

$ python hello_ray.py --no-useray
2021-06-25 20:31:16,733	INFO services.py:1272 -- View the Ray dashboard at http://127.0.0.1:8266
Not using ray
Took 0:00:10.023159 to return [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

$ python hello_ray.py --useray
2021-06-25 20:31:57,451	INFO services.py:1272 -- View the Ray dashboard at http://127.0.0.1:8266
Using ray
Took 0:00:01.092511 to return [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

To run this program on a remote cluster ensure that `ray.init(address="true")` is active and `ray.init()` is
commented out. Ensure that a remote cluster has been started

$ ray up aws-cluster/snippets.yml

$ ray submit aws-cluster/snippets.yml hello_ray.py -- --no-useray
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 03:33:24,005	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
Not using ray
Took 0:00:10.009363 to return [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
Shared connection to 52.10.109.67 closed.

$ ray submit aws-cluster/snippets.yml hello_ray.py -- --useray
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 03:32:59,926	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
Using ray
(autoscaler +4s) Tip: use `ray status` to view detailed autoscaling status. To disable autoscaler event messages, you can set AUTOSCALER_EVENTS=0.
(autoscaler +4s) Adding 1 nodes of type ray.worker.default.
Took 0:00:05.994300 to return [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
Shared connection to 52.10.109.67 closed.
"""

import time
from datetime import datetime

import click
import ray

# ray.init()
ray.init(address="auto")


def task(value):
    time.sleep(1)
    return value ** 2


@ray.remote
def task_with_ray(value):
    time.sleep(1)
    return value ** 2


@click.command()
@click.option("--useray/--no-useray", default=False)
def main(useray):
    start = datetime.now()
    if useray:
        print("Using ray")
        promises = [task_with_ray.remote(i) for i in range(10)]
        values = ray.get(promises)
    else:
        print("Not using ray")
        values = [task(i) for i in range(10)]
    end = datetime.now()
    print(f"Took {end-start} to return {values}")


if __name__ == "__main__":
    main()
