from asyncio import Event
from time import sleep
import ray
from tqdm import tqdm
import click
import random


@ray.remote
class ProgressBarActor:
    def __init__(self):
        self.counter = 0
        self.delta = 0
        self.event = Event()

    def update(self, num_items_completed):
        self.counter += num_items_completed
        self.delta += num_items_completed
        self.event.set()

    async def wait_for_update(self):
        await self.event.wait()
        self.event.clear()
        delta_ = self.delta
        self.delta = 0
        return delta_, self.counter

    def get_counter(self):
        return self.counter


class ProgressBar:
    def __init__(self, total, desc=""):
        self.progress_actor = ProgressBarActor.remote()
        self.total = total
        self.desc = desc

    @property
    def actor(self):
        return self.progress_actor

    def print_until_done(self):
        pbar = tqdm(desc=self.desc, total=self.total)
        while True:
            delta, counter = ray.get(self.actor.wait_for_update.remote())
            pbar.update(delta)
            if counter >= self.total:
                pbar.close()
                return


@ray.remote
def task(i, pba):
    n = random.random()
    sleep(n)
    pba.update.remote(1)
    return i


@click.command()
@click.option("--num-ticks", default=10, help="The number of ticks.")
def main(num_ticks):
    ray.init(address="auto")
    pb = ProgressBar(num_ticks)
    actor = pb.actor
    promises = [task.remote(i, actor) for i in range(num_ticks)]
    pb.print_until_done()
    vals = ray.get(promises)
    actual_num_ticks = ray.get(actor.get_counter.remote())
    print(f"Got {len(vals)} values")
    print(
        f"Expected number of ticks {num_ticks}, Actual number of ticks {actual_num_ticks}"
    )


if __name__ == "__main__":
    main()
