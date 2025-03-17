import os

import hydra
import torch.distributed as dist


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    os.environ["MASTER_ADDR"] = cfg.job.master_addr
    os.environ["MASTER_PORT"] = str(cfg.job.master_port)
    os.environ["RANK"] = str(cfg.job.rank)
    os.environ["WORLD_SIZE"] = str(cfg.job.world_size)
    print("Starting init_process_group..")
    dist.init_process_group("gloo")
    input("Press ENTER to exit:")
    print("Exiting.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
