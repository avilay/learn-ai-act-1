# TorchX

## Installation
The package does not exist on conda. Also, the official documentaiton asks to install `torchx[dev]` but that does not work.
`pip install torchx`



## Concepts

#### App Spec

The central concept here is that of an app spec. Think of this as similar to k8 spec. It defines the application that torchx runner should run. This is what an app spec that runs Python can look like -

```python
def python(...):
  return specs.AppDef(
    name="torchx_utils_python",
    roles=[
      specs.Role(
        name="python",
        image="ghcr.io/pytorch/torchx:latest",
        entrypoint="python",
        num_replicas=num_replicas,
        resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
        args=[*cmd, *args],
        env={"HYDRA_MAIN_MODULE": m} if m else {},
      )
    ],
  )
```

An app spec is really a function that returns a `spec.AppDef` dataclass. The way to to execute this app spec is to tell torchx to run this function, e.g., `torchx run python ...` along with whatever additional args that are needed.

The spec name and role name can be anything. The main section here is the `roles` section. Even though it is an array, so far I have not seen any app spec have more than one role. The role is where we specify the executable, the resource requirements, environment variables, arguments to the executable, etc. torchx is based on docker images. Every role needs to specify the image and the entrypoint to the image is essentially the executable. 

I can define my own app spec, but it should be rare. torchx comes with a number of pre-built app specs that I can just use, including the `python` app spec. These pre-built app specs are called *components* for some reason. Some useful app specs are `python`, `ddp`, etc. Each of these come with their own arguments that I can use to specify the resource requirements, etc.

Typically what I think of as an "app" is just the python package that I'll pass to one of the built-in components.

#### Scheduler

The main usecase of torchx is to launch any distributed workload on some distributed cluster. It supports a bunch of different schedulers and their corresponding clusters like K8, AWS Batch, ray, slurm, etc. Each scheduler has its own configuration options that we can provide to the torchx runner. For development purposes it also supports `local_docker` and `local-cwd`. `local_cwd` just takes the current working directory and copies it over to the default torchx docker container image and launches it.

#### Pipelines

TODO

## CLI

To list all the built-in apps -

```shell
torchx builtins
```



To list all supported schedulers and their corresponding arguments -

```shell
torchx runopts
```

To get info about a specific scheduler -

```shell
torchx runopts <schedulername>
```



To get help on running a particular component -

```shell
torchx run -s local_cwd dist.ddp --help
```



A really good way to get a better understanding of the component and the scheduler that will be used for a particular run is to use the `--dryrun` flag like so -

```shell
torchx run --dryrun -s local_cwd utils.python hello.py aptg
```



