# Next Steps	
1. Tutorials > PyTorch Lightning CIFAR10 ~94% Baseline Tutorial



## Misc Notes

The following tasks are avaialble in Lightning Flash as of today (12/31/2021):

* Image and Video
  * Image classification
  * Multi-label image classification
  * Image embedder
  * Object detection
  * Keypoint detection
  * Instance segementation
  * Semantic segmentation
  * Style transfer
  * Video classification
* Audio
  * Audio classification
  * Speech recognition
* Tabular
  * Tabular classification
  * Tabular forecasting
* Text
  * Text classification
  * Multi-label text classification
  * Question answering
  * Summarization
  * Translation
  * Text embedder
* Point cloud
  * Point cloud segementation
  * Point cloud object detection
* Graph
  * Graph classification
  * Graph embedder




These notes will need to be consolidated in proper topics later.

* `LightningModule`'s `forward` method is called during inference. It is not the full forward pass of my training. E.g., in an encoder/decoder setup, I’d probably need to do the forward pass of only my encoder, not my decoder in this method.
* Inside the `training_step`, `validation_step`, `test_step` methods, the way to log stuff is to call `self.log(“train_loss”, loss)`.  So probably something like `self.log(name, value)`.
* If my `LightningModule` has multiple models, say one encoder and one decoder module, then in the configure optimizer I don’t have to concatenate their parameters individually. I can just use `self.parameters()` and it will give me all the parameters of all the models that are defined in there.

## Common Use Cases

* I can create a child class of `Callback` and implement some (or all) of the hooks defined in the `Trainer`. Passing this callback class to the trainer will then execute the arbitrary code defined in my callback.




## Tutorials <–> Docs Mapping

| Tutorial                                      | Doc                                                          |
| --------------------------------------------- | ------------------------------------------------------------ |
| Automatic Batch Size Finder                   | [Common Use Cases > Training Tricks > Auto scaling of batch size](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#auto-scaling-of-batch-size) |
| Automatic Learning Rate Finder                | [Common Use Cases > Learning Rate Finder](https://pytorch-lightning.readthedocs.io/en/latest/advanced/lr_finder.html) |
| Exploding and Vanishing Gradients             | [Common Use Cases > Training Tricks > Gradient Clipping](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#gradient-clipping) |
| Lightning Early Stopping                      | [Common Use Cases > Early Stopping](https://pytorch-lightning.readthedocs.io/en/latest/common/early_stopping.html) |
| Lightning Weights Summary                     |                                                              |
| Lightning Progress Bar                        |                                                              |
| Lightning Profiler                            | [Common Use Cases > Performance and Bottleneck Profiler](https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html) |
| Controlling Lightning Training and Eval Loops | [Optional extensions > Loops](https://pytorch-lightning.readthedocs.io/en/latest/extensions/loops.html) |
| Training on GPUs                              | [Common Use Cases > Single GPU Training](https://pytorch-lightning.readthedocs.io/en/latest/common/single_gpu.html) |
| Training on Multiple GPUs                     | [Common Use Cases > Multiple GPU Training](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html) |
| Training on TPUs                              | [Common Use Cases > TPU Support](https://pytorch-lightning.readthedocs.io/en/latest/advanced/tpu.html) |
| Advanced Distributed Training                 |                                                              |
| Debugging Lightning Flags                     |                                                              |
| Accumulating Gradients                        | [Common Use Cases > Training Tricks > Accumulate gradients](https://pytorch-lightning.readthedocs.io/en/latest/advanced/training_tricks.html#accumulate-gradients) |
| Mixed Precision Training                      | [Common Use Cases > Mixed Precision Training](https://pytorch-lightning.readthedocs.io/en/latest/advanced/mixed_precision.html) |



## Docs Plan

| Path                                                         | Read | Order |
| ------------------------------------------------------------ | ---- | ----- |
| Common Use Cases > Cloud Training                            | N    |       |
| Common Use Cases > Computing cluster                         | Y    |       |
| Common Use Cases > Child Modules                             | Y    | B     |
| Common Use Cases > Debugging                                 | Y    | I     |
| Common Use Cases > Early Stopping                            | Y    | B     |
| Common Use Cases > Hyperparameters                           | Y    | B     |
| Common Use Cases >  Inference in Production                  | N    |       |
| Common Use Cases >  IPU support                              | N    |       |
| Common Use Cases > Lightning CLI and config files            | N    |       |
| Common Use Cases > Learning Rate Finder                      | Y    | B     |
| Common Use Cases > Loggers                                   | Y    | B     |
| Common Use Cases > Multi-GPU training                        | Y    | I     |
| Common Use Cases > Model Parallel GPU Training               | Y    | A     |
| Common Use Cases > Mixed precision Training                  | Y    | A     |
| Common Use Cases > Saving and loading weights                | Y    | B     |
| Common Use Cases > Fault-tolerant Training                   | Y    | I     |
| Common Use Cases > Custom Checkpoint IO                      | Y    | I     |
| Common Use Cases > Optimization                              | Y    | B     |
| Common Use Cases > Performance and Bottleneck Profiler       | Y    | I     |
| Common Use Cases > Training Type Plugins Registry            | Y    | I     |
| Common Use Cases > Remote filesystems                        | N    |       |
| Common Use Cases > Sequential Data                           | Y    | B     |
| Common Use Cases > Single GPU Training                       | Y    | B     |
| Common Use Cases > Training Tricks                           | Y    | B     |
| Common Use Cases > Pruning and Quantization                  | Y    | A     |
| Common Use Cases > Transfer Learning                         | Y    | I     |
| Common Use Cases > TPU support                               | N    |       |
| Common Use Cases > Test set                                  | Y    | B     |
| Optional extensions > Accelerators                           | Y    | I     |
| Optional extensions > Callback                               | Y    | B     |
| Optional extensions > LightningDataModule                    | Y    | B     |
| Optional extensions > Logging                                | Y    | B     |
| Optional extensions > Plugins                                | Y    | I     |
| Optional extensions > Loops                                  | Y    | A     |
| Tutorials > Step-by-step walkthrough                         | N    |       |
| Tutorials > PyTorch Lightning 101 class                      | N    |       |
| Tutorials > From PyTorch to PyTorch Lightning                | N    |       |
| Tutorials > Tutorial 1: Introduction to PyTorch              | N    |       |
| Tutorials > Tutorial 2: Activation Functions                 | N    |       |
| Tutorials > Tutorial 3: Initialization and Optimization      | Y    | I     |
| Tutorials > Tutorial 4: Inception, ResNet and DenseNet       | N    |       |
| Tutorials > Tutorial 5: Transformers and Multi-Head Attention | Y    | A     |
| Tutorials > Tutorial 6: Basics of Graph Neural Networks      | Y    | A     |
| Tutorials > Tutorial 7: Deep Energy-Based Generative Models  | Y    | A     |
| Tutorials > Tutorial 8: Deep Autoencoders                    | Y    | A     |
| Tutorials > Tutorial 9: Normalizing Flows for Image Modeling | Y    | A     |
| Tutorials > Tutorial 10: Autoregressive Image Modeling       | Y    | A     |
| Tutorials > Tutorial 11: Vision Transformers                 | Y    | A     |
| Tutorials > Tutorial 12: Meta-Learning - Learning to Learn   | Y    | A     |
| Tutorials > Tutorial 13: Self-Supervised Contrastive Learning with SimCLR | Y    | A     |
| Tutorials > GPU and batched data augmentation with Kornia and PyTorch Lightning | N    |       |
| Tutorials > Barlow Twins Tutorial                            | Y    | A     |
| Tutorials > Pytorch Lightning Basic GAN Tuotiral             | Y    | A     |
| Tutorials > PyTorch Lightning CIFAR10 ~94% Baseline Tutorial | Y    | A     |
| Tutorials > Pytorch Lightning DataModules                    | Y    | B     |
| Tutorials > Introduction to PyTorch Lightning                | N    |       |
| Tutorials > TPU training with PyTorch Lightning              | N    |       |
| Tutorials > How to train a Deep Q-Network                    | Y    | A     |
| Tutorials > Fintue Transformers Models with PyTorch Lightning | Y    | A     |



### Beginner Level

- [x] Common Use Cases > Loggers
- [x] Optional extensions > Logging
- [x] Common Use Cases > Saving and loading weights
- [x] Optional extensions > LightningDataModule
- [x] Common Use Cases > Test set
- [x] Common Use Cases > Hyperparameters
- [x] Common Use Cases > Single GPU Training
- [x] Common Use Cases > Training Tricks
  - [x] Accumulate Gradients
  - [x] Gradient Clipping
  - [x] Stochastic Weight Averaging
  - [x] Auto scaling of batch size
- [x] Common Use Cases > Child Modules
- [x] Common Use Cases > Early Stopping
- [x] Common Use Cases > Learning Rate Finder
- [ ] Common Use Cases > Sequential Data
- [x] Optional extensions > Callback



### Intermediate Level

- [x] Common Use Cases > Debugging
- [x] Common Use Cases > Multi-GPU training
- [x] Optional extensions > Accelerators
- [x] Optional extensions > Plugins
- [ ] Common Use Cases > Performance and Bottleneck Profiler
- [x] Common Use Cases > Transfer Learning
- [x] Common Use Cases > Training Type Plugins Registry
- [x] Tutorials > Tutorial 3: Initialization and Optimization
- [ ] Common Use Cases > Fault-tolerant Training
- [ ] ~~Common Use Cases > Custom Checkpoint IO~~



### Advanced Level

- [x] Optional extensions > Loops
- [ ] Common Use Cases > Optimization
- [ ] Common Use Cases > Model Parallel GPU Training
- [ ] Common Use Cases > Mixed precision Training
- [ ] Common Use Cases > Pruning and Quantization
- [x] Tutorials > PyTorch Lightning CIFAR10 ~94% Baseline Tutorial
- [ ] Tutorials > Tutorial 5: Transformers and Multi-Head Attention
- [ ] Tutorials > Tutorial 11: Vision Transformers
- [ ] Tutorials > Fintue Transformers Models with PyTorch Lightning
- [ ] Tutorials > Tutorial 6: Basics of Graph Neural Networks
- [ ] Tutorials > Tutorial 13: Self-Supervised Contrastive Learning with SimCLR
- [ ] Tutorials > Barlow Twins Tutorial
- [ ] Tutorials > Tutorial 8: Deep Autoencoders
- [ ] Tutorials > Pytorch Lightning Basic GAN Tutorial
- [ ] Tutorials > Tutorial 7: Deep Energy-Based Generative Models
- [ ] Tutorials > How to train a Deep Q-Network
- [ ] Tutorials > Tutorial 12: Meta-Learning - Learning to Learn
- [ ] Tutorials > Tutorial 9: Normalizing Flows for Image Modeling
- [ ] Tutorials > Tutorial 10: Autoregressive Image Modeling





