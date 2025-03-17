"""
Demonstrates that print statements are piped out to the local shell.

To run locally ensure that `ray.init()` is active and `ray.init(address="true")` is commented out.

$ python logray_1.py
2021-06-25 20:44:34,397	INFO services.py:1272 -- View the Ray dashboard at http://127.0.0.1:8266
(pid=39119) Inside task
(pid=39119) Working hard zzzzz...
(pid=39119) Done working.

To run on remote cluster ensure that `ray.init(address="true")` is active and the other init is commented out.

Ensure that the cluster is up.

$ ray up aws-config/snippets.yml

$ ray submit aws-cluster/snippets.yml logray_1.py
Loaded cached provider configuration
If you experience issues with the cloud provider, try re-running the command with --no-config-cache.
Fetched IP: 52.10.109.67
Fetched IP: 52.10.109.67
2021-06-26 03:48:25,495	INFO worker.py:734 -- Connecting to existing Ray cluster at address: 172.31.32.198:6379
(pid=23743) Inside task
(pid=23743) Working hard zzzzz...
(pid=23743) Done working.
Shared connection to 52.10.109.67 closed.
"""

import ray
import time

# ray.init()
ray.init(address="auto")


@ray.remote
def task():
    print("Inside task")
    print("Working hard zzzzz...")
    time.sleep(3)
    print("Done working.")


ray.get(task.remote())
