from termcolor import cprint
from colorama import init
import torch.distributed as dist

init()

rank_color = ["yellow", "green", "blue", "magenta", "red", "grey", "white", "cyan"]


def dist_print(text, *args, **kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    cprint(f"[{rank}]: {text}", rank_color[rank], *args, **kwargs)
