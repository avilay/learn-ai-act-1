# Results

## single-mnist
```
python single_mnist.py cmd=train
```
Takes around 1 minute per epoch. Val loss 0.221 and val accuracy is 0.935. Test accuracy of 0.942.

## dist-mnist
```
python dist_mnist.py cmd=train dist.rank=0
python dist_mnist.py cmd=train dist.rank=1
```
Took a total of 39 seconds for 3 epochs, so roughly 13 seconds per epoch. Val loss of 0.315 and val accuracy of 0.905.