"""
Demonstrates the use of a third party library and logging.
See the aws-cluster/snippets.yml on how to install third party python libraries.

To run locally ensure that `ray.init()` is active and `ray.init(address="auto")` is commented.

$ python logray_2.py
2021-06-25 22:57:35,647	INFO services.py:1272 -- View the Ray dashboard at http://127.0.0.1:8266
(pid=59185) [INFO 06-25 10:57:38] 59185-root: A log message from task
(pid=59191) [INFO 06-25 10:57:38] 59191-root: A log message from actor

To run on cluster.

$ ray submit aws-cluster/snippets.yml logray_2.py
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 06:21:54,546	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
(pid=27856) [INFO 06-26 06:21:55] 27856-root: A log message from actor
(pid=27879) [INFO 06-26 06:21:55] 27879-root: A log message from task
Shared connection to 52.10.109.67 closed.
"""

import ray
import logging
from snippets.log_config import configure_logger


ray.init(address="auto")
# ray.init()


@ray.remote
class Actor:
    def __init__(self):
        # Have to set the logger for each actor/task as they are
        # executed in separate python processes
        configure_logger()

    def log(self, msg):
        logging.info(msg)


@ray.remote
def f(msg):
    configure_logger()
    logging.info(msg)


actor = Actor.remote()  # type: ignore
actor_promise = actor.log.remote("A log message from actor")
ray.get(actor_promise)

task_promise = f.remote("A log message from task")
ray.get(task_promise)
