from yachalk import chalk
import torch.distributed as dist
import os

rank_color = [chalk.blue, chalk.magenta, chalk.yellow, chalk.green]


def dist_print(text, *args, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    pid = os.getpid()
    print(rank_color[rank](f"[{rank}]({pid}): {text}"), *args, **kwargs)
