# Multiple DataLoader Workers
The behavior is different for map-style and iterable-style datasets.

## Map-Style Datasets
The `DataLoader` instantiates the dataset in the main process and does some magic to copy the `__getitem__` method to different worker processes. In the main process it maintains a buffer of batches. By default the `prefetch_factor` for the data loader is 2, which means that it will try to hold 2*n_workers batches of rows in its buffer. As soon as the dataloader iterator is created, it starts sending requests to the worker processes to get a row. E.g., if there are 2 workers and the batch size is 3, the dataloader in the main process will send a request to w0 for [0], w1 for [3]. Once w0 gets back with [0], dataloader will send it a request for [1] and so on. It will keep doing this till it has $2 \times 2 = 4$ batches, i.e., it will make $3 \times 4 = 12$ requests to the two workers. Once it has 4 batches in its buffer it will stop making requests. When one batch is consumed it will make a corresponding requests to the next worker in line to get one full additional batch.

To see this in action run `multiworkers.py` as under:
```
python multiworkers.py
```

Sampling happens in the main process, once the main dataloader gets the order in which to get the indexes, it will start making the calls in a round-robin style, where the first worker will get all the indexes for the first batch, the second worker will get all the indexes for the second batch and so on.

## Iterable-Style Datasets
As before the dataloader instantiates the dataset in the main process and does some magic to copy the `__iter__` method to different worker processes and maintains a buffer of batches. The logic of filling the buffer is exactly as before. The difference is the absence of a sampler. So there are no indexes that are sent to the worker process. Only the `next` method is called in the worker process. So if the data is not sharded properly then it will be repeated by each worker process. In order to facilitate sharding there is a convenience function called `worker_init_fn` that I can pass to each worker. This can get the dataset instance that was passed to this process and set the right sharding value for the instance that belongs to that process. This means that my dataset must have some way to configure the shard that it should operate on. All the workers are still shipping their data to the same reader process.