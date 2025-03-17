# Ray

Any Python function that is decorated with `@ray.remote` will be executed in an async manner by the Ray runtime. These functions are called *tasks*. These functions then need to be called as `task.remote()` which will return a promise object. The task will start executing remotely. Calling `ray.get(promise)` is a blocking call and will return only when the task is complete.

==Is it possible to pickle the promise object and then get it from another process?==

To start the ray program locally I just run it like a normal python program `python script.py`. If I want to submit it to a cluster I'll have to `ray submit cluster.yml script.py`. Ray will use the `cluster.yml` to figure out which cluster to submit this job to. In any Ray program I need to call `ray.init()`. This is a weird function. This will only run locally. If I want to run the program on a remote cluster I'll have to use `ray.init(address="auto")`. Of course if I am already on the head node, then I can just run the program like any other python program with `ray.init()` and it will run on the cluster. 

One hack is to have a local cluster always running. That way all my programs can use `ray.init(address="auto")` and it will work both on the local and remote cluster. See `start_local.py` to start a local cluster.



To run the job on the remote cluster I need to have started the cluster first by calling - 

```
ray up cluster.yml
```

At the very least this will start a head node. If the config calls for it, it will also start a bunch of workers, but usually that is best left to autoscaling. To SSH into the cluster head node -

```
ray attach cluster.yml
```

The remote cluster has a web-based dashboard that is running on the head node. It can be mapped to a port on my local laptop by calling -

```
ray dashboard cluster.yml
```

Usually when I run a remote job, even though the job is running on the head node, the local ray process will pipe the head output to the local shell. So shutting down the local shell will kill the remote process. To run a long running job on the remote cluster I can run it in a detached tmux session on the head node as follows:

```
ray submit cluster.yml --tmux script.py
```

In order to check in on the job I can SSH into the head node and then attach to a tmux session. Ray offers a convenience method for this using the following command - 

```
ray attach cluster.yml --tmux
```

This will attach to the last tmux session.

