# Summary of Lightning APIs

## Lightning Flash

There is another library called [Lightning Flash](https://lightning-flash.readthedocs.io/en/latest/). It has a bunch of ML tasks implemented already and provides an even higher level API to use these tasks right away with little or no training. The following will create an image classifier with resnet18 (called *backbone* model) and the tack on a linear layer based on the number of classes specified (this part of the net is called the *head* model). The *freeze* strategy will freeze the backbone and only train the head.

```python
from pytorch_lightning import seed_everything

import flash
from flash.core.data.utils imoprt download_data
from flash.image import ImageClassificationData, ImageClassifier

seed_everything(42)

download_data("https://path/to/hymenoptera_data.zip", "/home/avilay/mldata")
data = ImageClassificationData.from_folders(
	train_folder="/home/avilay/mldata/hymenoptera/train/",
  val_folder="/home/avilay/mldata/hymeoptera/val/",
  test_folder="/home/avilay/mldata/hymenoptera/test",
  batch_size=32
)

model = ImageClassifier(backbone="resnet18", num_classes=data.num_classes)
trainer = flash.Trainer(max_epochs=10, gpus=-1)
trainer.fit(model, datamodule=data, strategy="freeze")
trainer.save_checkpoint("/path/to/classifier.ckpt")
```



## Overview

The three main classes in Lightning are `LightningModule`, `LightningDataModule`, and `Trainer`.

| `Trainer`                                    | `LightningModule`                                            | `LightningDataModule`                                        |
| -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `fit`, `validate`, `test`, `predict`, `tune` | `configure_optimizers`, `forward`, `predict_step`, `test_step`, `test_step_end`, `training_step`, `training_step_end`, `training_epoch_end`, `validation_step`, `validation_step_end`, `validation_epoch_end` | `prepare_data`, `setup`, `train_dataloader`, `val_dataloader`, `test_dataloader`, `predict_dataloader` |

The following methods on `LightningModule` must be implemented at a minimum:

* `configure_optimizers`
* `forward`
* `training_step`

### Training Loop

The `training_step` must return the loss. This can be returned directly or as part of a dict with the key `“loss”`. It is not necessary to call `forward` in the training loop. `forward` is only needed for inference. I’ll probably still do it to be compatible with pure PyTorch. The only use for `training_step_end` that I have found so far is to log metrics in a distributed training setting (see logging section below). However one of its input parameter is the output produced by the `training_step`. `training_epoch_end` can be used to log some more epoch level metrics, but even those can be logged in `training_step` with `on_epoch` flag set to `True`. One of its inputs is a list of outputs of all the steps in the epoch. If the output of `training_step` was a dict, this input argument will be an array of dicts. Neither `training_step_end` nor `training_epoch_end` need to return anything.

Calling `Trainer.fit` will call run the training loop and optionally the validation loop if it has been defined.

### Validation Loop

This is very similar to the training loop in that any output returned by `validation_step` is passed onto `validation_step_end` and `validation_epoch_end`. The only difference is that `validation_step` does not have to return any value.

If for some reason I want to only execute the validation loop on the entire validation batch I can call `Trainer.validate`.

### Test Loop

There are 4 ways I can test a model - 

1. Test a model that I just trained with the best checkpoint

```python
trainer.fit(model)
trainer.test(ckpt_path="best")
```

2. Test a model that I just trained with a specific checkpoint (not necessarily the best checkpoint)

```python
trainer.fit(model)
trainer.test(ckpt_path="/path/to/my/specific/ckeckpoint.ckpt")
```

3. Test a pre-trained model

```python
model = Net.load_from_checkpoint(checkpoint_path="/path/to/my/pretrained/model.ckpt")
trainer = Trainer()
trainer.test(model)
```

4. Test with a new data loader (applicable to all the above scenarios)

```python
trainer.test(dataloaders=test_dl)
# - OR -
test_dm = MyTestOnlyLightningDataModule()
trainer.test(datamodule=test_dm)
```

All of these require that I implement the `LightningModule.test_step` in my model.

### Inference

The `forward` method is used for inference. The `predict_step` is just a wrapper around it. However, `predict_step` can scale to multiple GPUs and multiple nodes so it is a better option to use it. I can use `forward` directly as well, but then I need to remember to set the model in eval mode, etc.

```python
class Net(pl.LightningModule):
  def forward(self, batch_x):
    return self.fc(batch_x)
  
model = Net()
model.eval()
with torch.no_grad():
  pred = model(inputs)
```

Alternatively I can use the `Trainer.predict` method that in turn will call the `predict_step` and so on. In the example below the predictor leverages both the GPUs to predict the given batch of data.

```python
class Net(pl.LightningModule):
  def forward(self, batch_x):
    return self.fc(batch_x)
    
  def predict_step(self, batch_x, batch_idx):
    return self(batch_x)
  
model = Net()
trainer = Trainer(gpus=2)
trainer.predict(model, data_module)
```



## Training Tricks

### Autoscaling batch size

To automatically find the biggest batch size that will fit in my GPU’s memory, I can use the `auto_batch_size_finder` flag. But according to the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#auto-scaling-of-batch-size) it only works if `train_dataloader` method is defined on my model. Not sure if it will still work if I am using `LightningDataModule`. It should..

### Gradient Clipping

Trainer has support for gradient clipping. Under the hood it is still calling `torch.nn.utils.clip_grad_norm()` or `torch.nn.utils.clip_grad_value()`. The difference between these two modes of gradient clipping is that in norm, the norm of the gradients of all the parameters is measured and the norm is clipped if beyond a certain threshold. In the value mode, each parameter’s gradient is measured against the threshold and clipped. Here is the API usage -

```python
# The default value is 0, which means no gradient clipping
# Setting the value to 0.5 means that the norm will always be kept <= 0.5
trainer = Trainer(gradient_clip_val=0.5)

# This will ensure that each parameter's gradient is kept <= 0.5
trainer = Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")
```

### Accumulate Gradients

The basic idea is that instead of $\theta \leftarrow \theta - \alpha \nabla_{\theta}J$ we do $\theta \leftarrow \theta - \alpha \sum_i^n \nabla_{\theta}J_i$. i.e., we accumulate the graidents over several batches before updating the parameters. The main motiviation to do this is if my optimal batch size is bigger than what will fit in my GPU memory, then instead of compromising with smaller batch sizes, I can accumulte the gradients from multiple batches before actually updating the parameters. Lightning lets me do this setting the number of batches to accumulate in the Trainer, `Trainer(accumulate_grad_batches=n)`.

### Stochastic Weight Averaging

Didn’t get the math or the intuition behind the basic idea. But the recipe is that I do 75% of the training in the usual way. Then in the last 25% of the time, I will set a constant (or a cyclical) learning rate and train over a bunch of epochs. In each epoch I update the model parameters as usual. At the end of each epoch I will save the model parameters somewhere and continue with the next epoch. After all the epochs are done, I’ll average the saved model parameters and this averaged value will become my final parameters. If I am using batch norm in my model, then I have to run one final epoch with the final averaged model so the batch norml layers can calculate their statistics. This is available in PyTorch and as a convenience option to the Lightning Trainer as well. The builtin [`StochasticWeightAveraging`](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.StochasticWeightAveraging.html#pytorch_lightning.callbacks.StochasticWeightAveraging) callback has some sensible defaults and it makes sense to try it out just to see if I see any gains.

```python
trainer = Trainer(callbacks=[StochasticWeightAveraging()])
```

More details can be found in this [PyTorch blog post](https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/).



## Learning Rate Finder

The LR finder will find the LR where the loss will slope down. It needs `self.learning_rate` in the `LightningModule` or have the corresponding fields in the hyper parameters that were saved, i.e, `self.hparams.learning_rate`. The recommended way to use this flag is as follows:

```python
trainer = Trainer(auto_lr_find=True)

# this will find the "best" learning rate and set it in my hparams
trainer.tune(model)

# this will use the learning rate that was set in my hparams by the tune step
trainer.fit(model)
```

Of course I should have something like this in my `LightningModule`-

```python
class Net(pl.LightningModule):
  def configure_optimizers(self):
    return t.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
```



## Optimizers

I need to override the [`pl.LightningModule.configure_optimizers()`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers) method to return the optimizer(s) and learning rate scheduler(s) that I want to use for my training. Here is the pseudocode for how a single optimizer and a single LRScheduler will be used by default -

```python
for epoch in range(n_epochs):
  for batch in traindl:
    optim.zero_grad()
    loss = self.training_step(batch, batch_idx)
    loss.backward()
    optim.step()  # optim is stepped every batch
  lr_sched.step()  # lr_sched is stepped every epoch
```

I can of course change when I step the LRScheduler and if I want to pass any metrics with the LRScheduler `step` method. If I want to do either of these things I need to use the `lr_scheduler_config` dict as specified in the doc above. Otherwise I can just return the LRScheduler object directly. 

If I want to use multiple optimizers and multiple LRSchedulers, the pseudocode will look like as follows:

```python
for epoch in range(n_epochs):
  for batch in traindl:
    for optim in optims:
      optim.zero_grad()
      loss = self.training_step(batch, batch_idx, optim_idx) 
      loss.backward()
      optim.step()
  
  for lr_sched in lr_scheds:
    lr_sched.step()
```

Multiple optimizers and schedulers are generally used when I am using multiple networks, e.g., in GANs, PPOs, etc. Typically each optimizer is used to optimizer a different network. I am guessing that in the `training_step` I can detect which optimizer is being used based on the `optim_idx` and then return the loss of that network.

This method can return a bunch of different things -

* A single optimizer

* List or tuple of optimizers

* Tuple of two lists, first list has one or more optimizer objects and the second list has -

  * one or more LRScheduler objects

  * one or more `lr_scheduler_config`s

* A single `dict` - lets call it **optimizer config** for the purposes of this note - with two keys - `optimizer` that has the optimizer object as its value, and `lr_scheduler` that has as its value -
  * a single LRScheduler object
  * a single `lr_scheduler_config`
* Tuple of multiple `dict`s as described above

Out of all of these I prefer to return -

1. A single optimizer object
2. A single optimizer config dict 
   1. with a standard LRScheduler object
   2. with a `lr_scheduler_config` for non-standard LRScheduler objects
3. A list of optimizer objects
4. A tuple of optimizer config dicts

#### Scenario 1: A single optimizer

Just return a single optimizer and it will be used as follows:

```python
def configure_optimizers(self):
  return Adam(...)
```

#### Scenario 2: A single optimizer, a single standard learning rate scheduler

I can either return a tuple with two lists, with the first list containing the optimizer and the second list containing the LR scheduler. But I prefer returning the dict because it is much more explicit.

```python
def configure_optimizers(self):
  optim = Adam(...)
  lr_scheduler = ExponentialLR(gamma=0.1)
  return {
    "optimizer": optim,
    "lr_scheduler": lr_scheduler
  }
```

#### Scenario 3: A single optimizer, a single non-standard learning rate scheduler

In the example below I am using the `ReduceLROnPlateau` scheduler that needs to give the metric to plateau on when step is called like so - `lr_scheduler.step(metric_val)`. In this case, use the `lr_schedueler_config` and set the `monitor` key. As long as this metric is logged using the `self.log()` method somewhere in the `pl.LightningModule` it will be automagically picked up and used when calling `step`.

==TODO: If instead of `self.log(“val_loss”, loss)` I use `self.log(“loss”, {“val_loss”: loss})` will this metric still be picked up automagically?==

```python
def configure_optimizers(self):
  optim = Adam(...)
  lr_scheduler = ReduceLROnPlateau(optim)
  return {
    "optimizer": optim,
    "lr_scheduler": {
      "scheduler": lr_scheduler,
      "monitor": "val_loss"
    }
  }

def validation_step(self, ...):
  ...
  self.log("val_loss", loss)
```

### Logging Learning Rates

I don’t need to explicitly call `self.log()` for this. I can simply add a callback to the `Trainer` and it will log the learning rates of the various LR schedulers that I have returned from `self.configure_optimizers()` method. If I want to give my learning rate metric a custom name I can set it in the `lr_scheduler_config`. 



## Early Stopping

There are two ways to do early stopping, the first one is useless, the second one uses the [`EarlyStopping`](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.EarlyStopping.html?highlight=EarlyStopping) built-in callback. This callback is called by the Trainer after the end of every validation epoch. This callback is called at the end of each validation epoch. It will look for the metric that was set in the `monitor` argument of this callback in the logs. It will average this value across all batches to get the value for the entire validation epoch. And then compare the aggregated value in the current validation epoch with the value in the last validation epoch and if it is worse, it will count as a bad run. After 3 bad runs, the training is stopped. The number of bad runs is controlled by the `patience` parameter. The `min_delta` decides what counts as a bad run. Its default value is 0, which means that as long as the metric does not degrade, even if it is the same as last time, it is considered a good run. If the metric to monitor is a loss metric like “val_loss” then the `mode` argument is set to `min`, its default value. I can choose to monitor some other metric like accuracy in which case I’ll have to set the mode to `max`. 

```python
class Net(pl.LightningModule):
  ::
  def validation_step(self, batch, batch_idx):
    :::
    self.log("val_loss", loss)
    
    
trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss")])
```

```python
class Net(pl.LightningModule):
  ::
  def validation_step(self, batch, batch_idx):
    :::
    self.log("val_acc", acc)
    

trainer = Trainer(callbacks=[EarlyStopping(monitor="val_acc", mode="max")])    
```



## Logging

By default Lightning uses Tensorboard to log metrics. The default location for the logs will be `os.getcwd()`. I have experimented with the `CSVLogger` and the `WandbLogger` as well. I can pass in the logger I want to use when I am initializing the `Trainer`. For Tensorboard logger, the logs are in the `lightning_logs` sub-directory of the current working directory. To see the logs do -

```shell
tensorboard --logdir=./lightning_logs/
```

The main idea is that Lightning will create a timeline graph of each metric that is logged. I can also include multiple metrics in the same chart. The metrics are displayed either at the end of a step (i.e., batch) or at the end of the epoch. 

### Step Level and Epoch Level Metrics

For epoch level metrics that were logged during a step-level hook, the logging sub-system will keep accumulating the metrics and will reduce the metrics across all ranks. Depending on which `LightningModule` hook is the log written in, the logging sub-system will figure out whether the metric is  step-level or an epoch level metric.

| Hook                                                         | Step Level | Epoch Level |
| ------------------------------------------------------------ | ---------- | ----------- |
| `on_train_start`, `on_train_epoch_start`, `on_train_epoch_end`, `training_epoch_end` | No         | Yes         |
| `on_before_backward`, `on_after_backward`, `on_before_optimizer_step`, `on_before_zero_grad` | Yes        | No          |
| `on_train_batch_start`, `on_train_batch_end`, `training_step`, `training_step_end` | Yes        | No          |
| `on_validation_start`, `on_validation_epoch_start`, `on_validation_epoch_end`, `validation_epoch_end` | No         | Yes         |
| `on_validation_batch_start`, `on_validation-batch_end`, `validation_step`, `validation_step_end` | No         | Yes         |

But I can always specify which level to log this in by passing in the `on_step` or `on_epoch` flags to the logging APIs. 

### Log APIs

Now for the actual logging APIs - there are two APIs and three different ways to log stuff.

The following will create a chart with a single graph in it for *metric_name*. 

```python
self.log("metric_name", metric_val)
```

The following will create a chart named *cost_metrics* with two graphs in it, one for *entropy* and another for *loss*.

```python
self.log("cost_metrics", {"entropy": e, "loss": l})
```

If I have to log a bunch of metrics then the following convenience API will still create a separate chart for each metric.

```python
self.log_dict({"metric_one": val1, "metric_two": val2})
```

### With Torchmetrics

This is a summary of the content in on [this page](https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html). The usage pattern is as follows:

```python
class Net(pl.LightningModule):
  def __init__(self):
    :::
    self.acc = tm.Accuracy()
  
  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.loss_fn(outputs, targets)
    return {"loss": loss, "outputs": outputs, "targets": targets}
    
  def training_step_end(self, retvals):
    outputs, targets = retvals["outputs"], retvals["targets"]
    self.acc(outputs, targets)
    self.log("train_acc_step", self.acc)
    
  def training_epoch_end(self, outs):
    self.log("avg_train_acc", self.acc)
    :::
```

Here are few things to note here - 

* Ordinarily I’d have to move the metrics to the correct device ([doc](https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices)), but if the metrics are defined inside the `LightningModule` then Lightning does this automatically for me ([doc](https://torchmetrics.readthedocs.io/en/latest/pages/lightning.html#torchmetrics-in-pytorch-lightning)). It is for this reason that in my own experiments I am better off defining the metrics along with the model. I tried to decouple this by defining the metrics inside a separate callback, but the headache of figuring out where/how Lightning initialized the metrics callback was not worth the effort. To see this failed experiment see `minified_examples/failed_experiment` directory. 
* I can pass the metrics object directly to `log` method, I don’t need to pull out the value.
* For epoch level metrics, I don’t need to call `acc.compute()` to get the aggregate. It is done automagically by `log`.
* Because I didn’t call `acc.compute()` I don’t have to call `acc.reset()` either at the end of the epoch. That too is done automagically by what I am assuming the `log` method.
* I don’t need to worry about the `sync` family of flags for the `log` method, torchmetrics automatically take care of all that.
* For single-mode training, I don’t need to log in `*_step_end()` methods, I can log directly in `_step()` methods.

### Logging Frequency

By default the logging subsystem will log the step level metrics at every 50 steps. To change this I can set the `log_every_n_steps=n` when initializing the Trainer. Further the logging sub-system will flush to disk every 100 steps. To change this behavior use `flush_logs_every_n_steps` setting when initializing the Trainer.

### Progress Bar

I can control how frequently the progress bar updates on the screen by setting the `Trainer(progress_bar_refresh_rate=n)`. It defaults to 1. If I set it to 0, the progress bar does not appear on the screen at all. By default the logging sub-system will show the loss values on the progress bar. If I want to show any other metric that I am logging, I can always set the `prog_bar` flag like under. This flag is turned off by default.

```python
self.log("metric_name", val, prog_bar=True)
```

If there is a metric that I only want to show on the progress bar but don’t want to create a graph for, I can turn off the `trainer` flag, which is turned on by default.

```python
self.log("metric_name", val, prog_bar=True, trainer=False)
```



## Saving and Loading Data

### Saving

This is controlled by the [ModelCheckpoint](https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) callback. By default the lightning sub-system will save the model after the last epoch is complete in the current working directory. This callback with the default config is added automatically to the Trainer, so I don’t have to do anything. This will save the following:

* Current epoch
* Global step
* Model state_dict
* State of all optimizers
* State of all learning rate schedulers
* State of all callbacks
* Hyperparameters

There are ways I can specify that I only want to save weights and not all of this stuff, but I don’t think I’ll be using that too frequently. In case I want to customize the behavior of this callback I can create my own instance and pass it in. Here are some of the common changes in the setup -

```python
class Net(pl.LightningModule):
  def validation_step(self, batch, batch_idx):
    :::
    self.log("val_loss", val_loss)
    

ckpt_cb = ModelCheckpoint(
  monitor="val_loss",
  dirpath="my/path",  # relative to the default_root_dir in Trainer
  filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
  every_n_epochs=5,
  mode="min"  # this is the default value
)  
trainer = Trainer(
  default_root_dir="/path/to/rootdir"
  callbacks=[ckpt_cb]
)
print(f"Best checkpoint available at: {ckpt_cb.best_model_path}")
```

If I set the `monitor` setting with some custom metric then the best model will be saved. ==Will `every_n_epochs` disable saving the best model?== I can disable checkpointing entirely by setting `checkpoint_callback=False`. For DDP training this automatically saves the model only on rank:0 trainer.

#### Hyperparameters

The checkpointing sub-system will automatically save the hyperparameters that I have passed to the `LightningModule`. The hyperparameters can be accessed as follows -

```python
ckpt = torch.load(ckpt_path)
print(ckpt["hyperparameters"])
```

#### Manual

Use the `save_checkpoint` method on the Trainer. In the DDP setting, this will automatically save the model only on the rank:0 trainer.

```python
trainer = Trainer(strategy="ddp")
model = Net(hparams)
trainer.fit(model)
# Saves only on the main process
trainer.save_checkpoint("example.ckpt")
```

### Loading

The easiest way to restore the full checkpoint and resume training from there is to call the `Trainer.fit` method with the checkpoint path. I think this path is relative to the default root working dir.

```python
model = Net()
trainer = Trainer()
trainer.fit(model, ckpt_path="some/path/to/my/checkpoint.ckpt")
```

#### Manual

If I am not going to use the Trainer, e.g., when I want to do evaluation, then -

```python
model = Net.load_from_checkpoint("/full/path/to/my/checkpoint.ckpt")
model.eval()
preds = model(inputs)
```

In the DDP setting, I do have to map the location of the tensors from `cuda:0` to whatever device the trainer is running on.

```python
Net.load_from_checkpoint(
    'path/to/checkpoint.ckpt',
    map_location={'cuda:1':'cuda:0'}
)
```



## Lightnin Data Module

The `LightningDataModule` interface has the following methods -

#### `prepare_data` ,  `setup`, and `teardown`

These two methods are used to create the dataloaders or at the very least build the datasets. The difference between them shows up in the DDP settings. `prepare_data` is typically called from the local rank 0 trainer and it does one time things like downloading the datasets. If there are multiple hosts (or nodes) then `prepare_data` will be automatically called on each hosts’s local rank 0 trainer.  `setup` is called on each device, so really on each local trainer, and this can be used to actually prepare the data loaders. For this reason, it is recommneded that we don’t set state in `prepare_data`. Setting state in `setup` is fine.

`teardown` is used to undo stuff done in `setup`. This is also called on all the trainers.

#### Dataloaders

The methods are `train_dataloader`, `val_dataloader`, `test_dataloader`, `predict_dataloader`. These are called by the Lightning Trainer to get the appropriate data loader. For non-distributed training I do need to set the `shuffle` flag on my dataloaders as I normally would. However when I am overfitting the model, Lightning will turn off `shuffle` flag even if it is set.

#### Other hooks

There are a bunch of other event callbacks. For more details see the [LightningDataModule doc](https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html)

#### Stages

A bunch of APIs accept a string as `stage`, this can be one of the following - `fit`, `validate`, `test`, or `predict`.



## Hyperparameters

Hyperparameters are passed in to both `LightningModule` as well as `LightningDataModule`. By default any arguments passed into the `__init__` method are considered hyperparameters. The call to `self.save_hyperparameter()` will make these hyperparameters available in `self.hparams` collection. These hyperparameters are also logged if the logger being used supports it. When taking a checkpoint they are automatically saved. When restoring from a checkpoint they are available in the `checkpoint[“hyperparameteres”]` key. Instead of saving all the `__init__` args, I can also save a specific object as my hyperparameters, this can be a `dict`, `NameSpace`, or `OmegaConf`. `NameSpace` is Lightning’s support for CLIs which I’ll most probably not end up using. Here is the typical flow for me -

```python
class Net(pl.LightningModule):
  def __init__(self, hp: OmegaConf):
    self.save_hyperparameters(hp)
    
  def configure_optimizers(self):
    return Adam(lr=self.hparams.learning_rate)
  :::
        
        
model = Net.load_from_checkpoint("/path/to/checkpoint.ckpt")
model.hparams  # Will have the hyperparams automatically loaded from the restored model
```



## Debugging

Lightning has some nifty debugging flags.

### Run only a few batches

Often when developing a model iteratively it makes sense to run the model on a few batches to ensure the software engineering part is correct. There are three ways to do this in Lightning -

1. Run a single batch from train, val, and test data loaders and then exit. I can even set this flag to an integer and it will run that many batches. This is only really useful to sanity check the `training_step` and `validation_step` along with the dataloader. It disables checkpoints, early stopping, logging, etc.

   ```python
   trainer = Trainer(fast_dev_run=True)
   ```

2. Shorten epochs by training on only a subset of the data, but the full training is exercised.

   ```python
   # use only 10% of training data and 1% of val data
   trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)
   
   # use 10 batches of train and 5 batches of val
   trainer = Trainer(limit_train_batches=10, limit_val_batches=5)
   ```

3. Validation sanity check: By default Lightning will run 2 steps of `validation_step` at the beginning of the training. I can control the number of steps in the `num_sanity_val_steps` setting of the `Trainer`.

### Model Overfit

A good modeling technique is to massively overfit the model on a small subset of the training data. If the model has poor results on that, it means that there is something wrong with my model. Lightning has a way to use the same subset of training data and use it for training, validation, and testing. It also turns off the `shuffle` flag in my dataloader for even more aggressive overfitting.

```python
# use only 1% of training data (and turn off validation)
trainer = Trainer(overfit_batches=0.01)

# similar, but with a fixed 10 batches
trainer = Trainer(overfit_batches=10)
```

### Advanced Logging

1. Log the collective norm (usually makes sense to log the L2 norm) of the gradients of all the parameters of the model to see if any learning is happening.

   ```python
   # the 2-norm
   trainer = Trainer(track_grad_norm=2)
   ```
   
2. Log the device statistics. Useful when using multiple GPUs and I want to know the load on each. I think behind the scenes it polls `nvidia-smi` to get the stats so can be slow.
   
   ```python
   from pytorch_lightning.callbacks import DeviceStatsMonitor
   
   trainer = Trainer(callbacks=[DeviceStatsMonitor()])
   ```
   
   

## GPU Training

### Illustrative Example

1. Ensure there are no calls to `.to(device)` or `.to(cuda)` in my code. ==Is it ok to have `.to(t.float32)` in there?==
2. Ensure that the dataloader that I return does not have any samplers set.
3. Initialize my `Trainer` with the number of GPUs I want to use and the distributed strategy.
4. Ensure my code has a main entrypoint if I am using `ddp` as my strategy.

```python
# There are no calls to .to(device) or .to(cuda)
class Net(pl.LightningModule):
  def __init__(self):
    self.fc = nn.Linear(28*28, 1)
    self.loss_fn = nn.BCELoss()
    
  def forward(self, batch_x):
    y_hat = t.sigmoid(self.fc(batch_x))
    return t.squeeze(y_hat, dim=1)	
  
  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    loss = self.loss_fn(outputs, targets.to(t.float32))
    return loss

# I don't need to set the DistributedSampler, I don't even have to shuffle the dataset  
class MyData(pl.LightningDataModule):
  def train_dataloader(self, stage):
    if stage == "fit":
      return DataLoader(self.trainset, batch_size=self.hparams.batch_size)


def main():
  model = Net()
  dm = MyData()
  trainer = Trainer(gpus=-1, strategy="ddp")
  trainer.fit(model, dm)
  

if __name__ == "__main__":
  main()
```

### Hardware agnostic code

No need to explicitly move a tensor or a model to a specific device using `tensor.to(device)` or `tensor.cuda()`. Lightning will automatically figure out which device the tensor is supposed to be on and will move it there. If I am creating a new tensor in any of the Lightning hooks then I should use `new_tensor.type_as(old_tensor)` or `self.register_buffer(“sigma”, torch.eye(3))`. 

### Remove Samplers

For DDP training Lightning will automatically take the data loader returned by my data module and retro-fit it with a `DistributedSampler`. I don’t have to do it in my code. Of course if I want to use a custom sampler or want to stop Lightning from messing with my dataloader I can turn off `replace_sampler_ddp` flag in the `Trainer`. By default it is set to `True`.

### Synchronized Logging

As long as I am logging `Torchmetrics` I don’t have to do anything special. For other things I need to either sync the trainers before logging, or log only on the rank:0 trainer.

```python
# might have a performance impact
def validation_step(self, batch, batch_idx):
  :::
  self.log("my_custom_metric", val, on_step=True, on_epoch=True, sync_dist=True)
  
# have only rank:0 log
def test_step(self, batch, batch_idx):
  :::
  if self.trainer.is_global_zero:
  	self.log("my_reduced_metric", redval, rank_zero_only=True)
```

### Select GPUs to use

```python
# Use 3 GPUs per node
Trainer(gpus=3)
Trainer(gpus=[0, 1, 2])  # equivalent to above
Trainer(gpus=3, auto_select_gpus=True)  # Will choose any unoccupied 3 GPUs on my node

# Use only these GPUs	
Trainer(gpus=[1, 3])

# Use all available GPUs
Trainer(gpus=-1)

# Don't use any GPUs, do training on CPUs
Trainer(gpus=0)
```

By default Lightning will use the NCCL backend when running on GPUs.

### Select the training strategy

There are a bunch of different training strategies listed in the documentation page, but really I’ll be working with either `ddp` or `ddp_spawn`. There was a writeup somewhere that explained the difference between the two, but I cannot find it now. From what I remember, `ddp` means that Lightning will automatically launch the main script like this -

```shell
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=0 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=1 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
MASTER_ADDR=localhost MASTER_PORT=random() WORLD_SIZE=3 NODE_RANK=2 LOCAL_RANK=0 python my_file.py --gpus 3 --etc
```

==Do I have to use Lightning.CLI for this?== I don’t think so. I think the logic is that it will take whatever args I manually launch the script with, and then re-launch the script with the additional env variables along with the same args. But need to verify this.

**I think** `ddp_spawn` means that it is using the `torch.multiprocessing` library to create a different process that just executes the specified function. So probably it is spawing the `Trainer.fit` method in a separate sub-process. If for some reason my Python package does not have a `__main__` entry point, then I can use `ddp_spawn`. This is usually true when I am launching via Jupyter notebooks.

More info in [Graphics Processing Unit (GPU) — PyTorch Lightning 1.6.0dev documentation (pytorch-lightning.readthedocs.io)](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu.html#distributed-modes)



## Custom Loops

For reference, here is what a training loop without Lightning looks like -

```python
for epoch in range(max_epochs):
  	model.train()
    with t.enable_grad():
			for batch_idx, batch in enumerate(traindl):
  			inputs, targets = batch
  		
      	# standard 5-step process
  			optim.zero_grad()
  			outputs = model(inputs)
  			loss = loss_fn(outputs, targets)
  			loss.backward()
  			optim.step()
    
    model.eval()
    with t.no_grad():
      for batch_idx, batch in enumerate(valdl):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        # log the validation loss
```

This is what my conceoptual understanding of how Lightning splits this loop -

```python
for epoch in range(max_epochs):  # in Trainer
  	model.train()   # in Trainer
    with t.enable_grad():  # in Trainer
			for batch_idx, batch in enumerate(traindl):  # in Trainer
  			inputs, targets = batch  # impl by me in training_step
  		
      	# standard 5-step process
  			optim.zero_grad()  # in Trainer
  			outputs = model(inputs)  # impl by me in training_step
  			loss = loss_fn(outputs, targets)  # impl by me in training_step
  			loss.backward()  # in Trainer
  			optim.step()  # in Trainer
    
    model.eval()  # in Trainer
    with t.no_grad():  # in Trainer
      for batch_idx, batch in enumerate(valdl):  # in Trainer
        inputs, targets = batch  # impl by me in validation_step
        outputs = model(inputs)  # impl by me in validation_step
        loss = loss_fn(outputs, targets)  # impl by me in validation_step
        # log the validation loss  impl by me in validation_step
```

Lightning has abstracted this logic in a into an abstract base class called [`Loop`](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.loops.base.Loop.html#pytorch_lightning.loops.base.Loop). If I look at the `Trainer` class, it defines these loops - 

```python
class Trainer:
  def __init__(self):
    fit_loop = FitLoop(...)
    training_epoch_loop = TrainingEpochLoop(...)
    fit_loop.connect(epoch_loop=training_epoch_loop)
    self.fit_loop = fit_loop
    
    self.validate_loop = EvaluationLoop()
    self.test_loop = EvaluationLoop()
    self.predict_loop = PredictionLoop()
```

 Each of `FitLoop`, `TrainingEpochLoop`, `EvaluationLoop`, and `PredictionLoop` is a concrete child class of the `Loop` ABC. 

If I follow the `Trainer.fit` method, eventually I’ll end up in `_run_train` which does the following -

```python
def _run_train(self):
  self.model.train()
  torch.set_enable_grad(True)
  self.fit_loop.run()
```

The `run` method is defined in the `Loop` ABC and is conceptually implemented as follows -

```python
def run(self):
  self.on_run_start()
  while not self.done:
    self.on_advance_start()
    self.advance()
    self.on_advance_end()
  return self.on_run_end()
```

Most of these concrete child classes use the base class `run` method, which is the main entry point into any `Loop` class. If I follow the rabbit hole of `FitLoop` here is the hierarchy of loops -

```python
class FitLoop(Loop):
  def __init__(self):
    self.epoch_loop = TrainingEpochLoop()
    
  def advance(self):
    self.epoch_loop.run()
    
    
class TrainingEpochLoop(Loop):
  def __init__(self):
    self.batch_loop = TrainingBatchLoop()
    self.val_loop = EvaluationLoop()
  
  def advance(self):
    batch_idx, batch = next(self._dataloader_iter)
    self.batch_loop.run(batch, batch_idx)
    
  def on_advance_end(self):
    self.val_loop.run()
    
    
class TrainingBatchLoop(Loop):
  def __init__(self):
    # These two don't have any more sub-loops
		self.optimizer_loop = OptimizerLoop()
  	self.manual_loop = ManualOptimization()
  
class EvaluationLoop(DataLoaderLoop):
	  def __init__(self):
      # This does not have any sub-loops
      self.epoch_loop = EvaluationEpochLoop()      
```

The `TrainingBatchLoop` will call either the `self.optimizer_loop` or the `self.manual_loop` depending on whether manual loop has been defined. It is still not clear to me where the epcohs are set up and how does the validation loop in the `Trainer` get to the `TrainingEpochLoop`. 

### Wiring up custom loops

I can always create a fully custom loop and wire it up with the `Trainer` so that it will use my custom loop instead of the `FitLoop`. At the very least I need to implement `done` and `advance` and then set it in the `Trainer`.

```python
class MyFancyLoop(Loop):
  @property
  def done(self):
    :::
          
  def advance(self):
    :::
          
trainer = Trainer()
trainer.fit_loop = MyFancyLoop()
```

Or, I can subclass `FitLoop` and only customize parts of it -

```python
class CustomFitLoop(FitLoop):
  def advance(self):
    :::
          
trainer = Trainer()
trainer.fit_loop = CustomFitLoop()
```

If I want to customize only parts of one of the nested loops then this is the pattern -

```python
class MyEpochLoop(TrainingEpochLoop):
  :::
        
trainer.fit_loop.replace(epoch_loop=MyEpochLoop)
trainer.fit(model)   # will use the Custom training epoch loop
```

For some reason I should not just do -`trainer.fit_loop.epoch_loop = Custom()`. Not sure why. Another way to wire up custom loops is to use the `connect` method, where I can keep the next level nested loops and just customize the behavior of one of the intermediate nested loop -

```python
epoch_loop = MyEpochLoop()
epoch_loop.connect(trainer.fit_loop.epoch_loop.batch_loop, trainer.fit_loop.epoch_loop.val_loop)
trainer.fit_loop.connect(epoch_loop=epoch_loop)
trainer.fit(model)
```



## Plugins and Accelerators

There are four concepts here - 

1. Strategy plugins
2. Precision plugins
3. Accelerators
4. Clusters

Strategies are different training strategies like DDP, Horovod, DeepSpeed, etc. All strategies are derived from the ABC [`Strategy`](https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/strategies/strategy.html#Strategy). A strategy takes in references to a precision plugin and an accelerator each. 

```python
class Strategy(ABC):
  def __init__(self, accelerator=None, precision_plugin=None, ...):
    :::
```

Precisions are different precision strategies like MP, sharded MP, double precision, etc. Again, there is a precision base class that all precision plugins need to derive from.

Accelerators are for hardware specific setup of environments. There are a handful of accelerators today like CPU, TPU, GPU, and IPU. Again, there is a base class called `Accelerator` that all accelerators need to derive from.

It is not clear how clusters fit into all of this.

There are some flags that when passed to the `Trainer` will setup the right strategy, precision, and accelerator. e.g., 

```python
# accelerator: GPUAccelerator
# training strategy: DDPStrategy
# precision: NativeMixedPrecisionPlugin
trainer = Trainer(gpus=4, precision=16)
```

But I can always explicitly set these up -

```python
trainer = Trainer(strategy=CustomDDPStrategy(), plugins=[CustomPrecisionPlugin()])
```

It seems that the `plugins` param also accepts strategy classes, so I guess I could’ve passed in the `CustomDDPStrategy` in the `plugins` array?

And finally, I can create my own strategy with custom precision and custom hardware accelerators.

```python
trainer = Trainer(
  strategy=CustomDDPStrategy(
    accelerator=MyAccelerator(),
    precision_plugin=MyPrecision()
  )
)
```

There is a plugins registry where all known plugins have some sort of string name:

```python
from pytorch_lightning.plugins import TrainingTypePluginsRegistry

for plugin in TrainingTypePluginsRegistry.keys():
    print(plugin)
```

This will output -

```
fsdp
ddp_find_unused_parameters_false
ddp_sharded_find_unused_parameters_false
ddp_spawn_find_unused_parameters_false
ddp_sharded_spawn_find_unused_parameters_false
deepspeed
deepspeed_stage_1
deepspeed_stage_2
deepspeed_stage_2_offload
deepspeed_stage_3
deepspeed_stage_3_offload
deepspeed_stage_3_offload_nvme
tpu_spawn_debug
```



## Callbacks

These are a pretty nifty way of adding more functionality to the model training without polluting the main module or the trainer. The base callback class has a bunch of hooks that my derived callback can implement, e.g., `on_training_start`, `on_validation_start`, `on_train_batch_start`, etc. Lighthing will call any implemented hooks during training. These seem like a good place to stuff my metrics calculation and logging instead of putting that logic in my module.



## Transfer Learning

### Recap pure PyTorch

Lets take resnet18, it can be seen that this class has an attribute called `fc` is a linear layer and it is trained for 1000 classes.

```
model = tv.models.resnet18(pretrained=True)
model

ResNet(
  (conv1): ...
  (bn1): ...
  (relu): ...
  (maxpool): ...
  (layer1): Sequential(...)
  (layer2): Sequential(...)
  (layer3): Sequential(...)
  (layer4): Sequential(...)
  (avgpool): ...
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

I can do transfer learning in the following two ways -

* End-to-end training
  * Load the pre-trained weights.
  * Swap the last `fc` layer with another FC layer that has the requisite amount of out features equal to the number of classes in my problem domain.
  * Do training as usual.
* Fine-tuning
  * Load the pre-trained weights.
  * Swap the last `fc` layer as before.
  * Freeze all but the last layer.
  * Configure the optimizer to only optimize the newly created FC layer.
  * Train loop

#### End-to-End Training

```python
model = tv.models.resent18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, n_classes)
optim = t.optim.SGD(model.parameters(), lr=0.001)
# training loop
```

#### Finetuning

```python
model = tv.models.resnet18(pretrained=True)

# Freeze all the layers
for param in model.parameters():
  param.requires_grad = False
  
# All newly created modules have requires_grad set to True by default  
model.fc = nn.Linear(model.fc.in_features, n_classes) 

# Configure optimizer with the last layer's parameters
optim = t.optim.SGD(model.fc.parameters(), lr=0.001)

# training loop
```

### Lightning

`LightningModule` has a method called `freeze` that sets the `requires_grad` to the model’s parameters to `False` and set the model in evaluation model with `self.eval()`.

#### End-to-End Training

Not a lot changes -

```python
class ImageClassifier(LightningModule):
  def __init__(self):
    super().__init__()
    self.model = tv.models.resnet18(pretrained=True)
    self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
    self.loss_fn = ...
    
	def configure_optmizers(self):
    return t.optim.SGD(self.model.parameteres(), lr=0.001)
  
  def forward(self, inputs):
    return self.model(inputs)
  
  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    return self.loss_fn(outputs, targets)
```

#### Finetuing

A couple of different ways to do this - 

```python
class ImageClassifier(LightningModule):
  def __init__(self, n_classes):
    super().__init__()
    self.model = tv.models.resnet18(pretrained=True)
    for param in self.model.parameters():
  		param.requires_grad = False
    self.model.fc = nn.Linear(model.fc.in_features, n_classes)
    self.loss_fn = ...
    
  def configure_optmizers(self):
    return t.optim.SGD(self.model.fc.parameters(), lr=0.001)
    
  def forward(self, inputs):
    return self.model(inputs)
  
  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    targets = self(inputs)
    return self.loss_fn(outputs, targets)    
```



```python
class ImageClassifier(LightningModule):
  def __init__(self, n_classes):
    super().__init__()
    backbone = tv.models.resnet18(pretrained=True)
    n_features = backbone.fc.in_features
    layers = list(backbone.children())[:-1]
    self.feature_extractor = t.nn.Sequential(*layers)
    self.classifier = t.nn.Linear(n_features, n_classes)
    self.loss_fn = ...
    
  def configure_optimizers(self):
    return t.optim.SGD(self.classifier.parameters(), lr=0.001)
    
  def forward(self, inputs):
    self.feature_extractor.eval()
    with t.no_grad():
      repr = self.feature_extractor(inputs).flatten(1)
    return self.classifier(repr)
  
  def training_step(self, batch, batch_idx):
    inputs, targets = batch
    outputs = self(inputs)
    return self.loss_fn(outputs, targets)
```

If the backbone model is a `LightningModule` then I can call `freeze` method on it to freeze the params.

