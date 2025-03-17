New Concepts
============
How to standardize an image?
----------------------------
For non-image data, standardization means taking the entire column of training data, finding its mean and stddev and normalizing the column. In the context of an image does this mean we take position 0,0,0 and normalize the pixels in this position of all images? Or does it mean that we consider each image on its own an take the mean and stddev of all the images in that image?

What is Batch Normalization?
----------------------------

What is ResNET?
---------------
A great explanation of ResNETs, HighwayNets, and DenseNets along with simple(?) tensorflow implementations of all.
[Link](https://chatbotslife.com/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32)

Original MS Research paper on ResNets. Look in papers.

Tutorial on ResNet by Kaiming He
[Link](http://kaiminghe.com/icml16tutorial/icml2016_tutorial_deep_residual_networks_kaiminghe.pdf)

RNN
---
http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/
http://course.fast.ai/lessons/lesson6.html
http://yerevann.com/a-guide-to-deep-learning/
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

Sharpen Existing Concepts
=========================
A great summary (with the right amount of detail) on various gradient descent methodologies.
[Link](http://sebastianruder.com/optimizing-gradient-descent/index.html)

How to choose the batch size for SGD? For large batches I get poor performance, but for smaller batches I get better performance. This SE question goes into the math behind choosing the batch size.
[Link](http://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent)

Analysis of dropout. I also have the original research paper where this idea was first introduced. I am guessing this web page will be easier to digest than that paper.
[Link](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)


