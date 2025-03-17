## Create a Neural Net
There are three ways to create a network:

  1. Create a class that derives from nn.Module and implement the `forward` method.
  2. Use nn.Sequential to pass in an OrderedDict of layers.
  3. Use nn.ModuleList to hold the layers in a list. Usually used with the class based approach.

The preferred way is to create a class. The `__init__` method in the class is only used to pre-define certain kinds of layers. But the real implementation of the network structure is in the `forward` method. Because Sequential cannot reshape or flatten any layers, it is customary to use Sequential either when there is no need for reshaping, or define a number of different Sequential layers in the class.

## Different Modes
### Training
When training set the `model.train()` mode. This will enable dropouts (among other things).
```
model.train()
for epoch in range(epochs):
    :
    output = model.forawrd(...)
    :
```

### Inference
When doing inference we need to disable dropouts and there is no need to keep track of the gradients. Two things to do at this point are to set the evaluation of the model using `model.eval()` and turn off autograd by `torch.no_grad()`

```
model.eval()
with torch.no_grad():
    output = model.forward(...)
    ...
```

This disables the autograd feature that takes longer to compute.

## Typical Train Flow

```
model = MyModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

model.train()
for epoch in range(epochs):
    for batch in trainloader:
        inputs, targets = batch
        with torch.enable_grad():
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
```

## Typical Test Flow

```
model.eval()
with torch.no_grad():
    for batch in testloader:
        inputs, targets = batch
        outputs = model.forward(inputs)
```

## Typical Single Instance Inference Flow

```
model.eval()
with torch.no_grad():
    batch_of_one = input.view(1, input.size())
    predicted = model.forward(batch_of_one)[0]
```

# Popular Models

## LENET
#### Layer 1
32x32x1 -- CONV(k=5 c=6) --> 28x28x6 -- AVGPOOL(2) --> 14x14x6

#### Layer 2
14x14x6 -- CONV(k=5 c=16) --> 10x10x16 -- AVGPOOL(2) --> 5x5x16

#### Layer 3
5x5x16 -- FLATTEN --> 400 -- LINEAR --> 120

#### Layer 4
120 -- LINEAR --> 84

#### Layer 5
120 -- LINEAR -- SOFTMAX --> 10


## ALEXNET
#### Layer 1
227x227x3 -- CONV(k=11 s=4 p=2) -- RELU --> 56x56x64 -- MAXPOOL(k=3 s=2) --> 27x27x64

#### Layer 2
27x27x64 -- CONV(k=5 p=2) -- RELU --> 27x27x192 -- MAXPOOL(k=3 s=2) --> 13x13x192

#### Layer 3
13x13x192 -- CONV(k=3 p=1) -- RELU --> 13x13x384

#### Layer 4
13x13x384 -- CONV(k=3 p=1) -- RELU --> 13x13x256

#### Layer 5
13x13x256 -- CONV(k=3 p=1) -- RELU --> 13x13x256 -- MAXPOOL(k=3 s=2) --> 6x6x256

#### Layer 6
6x6x256 -- FLATTEN --> 9216 -- DROPOUT -- LINEAR -- RELU --> 4096

#### Layer 7
4096 -- DROPOUT -- LINEAR -- RELU --> 4096

#### Layer 8
4096 -- LINEAR --> 1000


## VGG 16
#### Layer 1
224x224x3 -- CONV(k=3 p=1) -- RELU --> 224x224x64

#### Layer 2
224x224x64 -- CONV(k=3 p=1) -- RELU --> 224x224x64 -- MAXPOOL(2) --> 112x112x64

#### Layer 3
112x112x64 -- CONV(k=3 p=1) -- RELU --> 112x112x128

#### Layer 4
112x112x128 -- CONV(k=3 p=1) -- RELU --> 112x112x128 -- MAXPOOL(2) --> 56x56x128

#### Layer 5
56x56x128 -- CONV(k=3 p=1) -- RELU --> 56x56x256

#### Layer 6
56x56x256 -- CONV(k=3 p=1) -- RELU --> 56x56x256

#### Layer 7
56x56x256 -- CONV(k=3 p=1) -- RELU --> 56x56x256 -- MAXPOOL(2) --> 28x28x256

#### Layer 8
28x28x256 -- CONV(k=3 p=1) --> 28x28x512

#### Layer 9
28x28x512 -- CONV(k=3 p=1) --> 28x28x512

#### Layer 10
28x28x512 -- CONV(k=3 p=1) --> 28x28x512 -- MAXPOOL(2) --> 14x14x512

#### Layer 11
14x14x512 -- CONV(k=3 p=1) --> 14x14x512

#### Layer 12
14x14x512 -- CONV(k=3 p=1) --> 14x14x512

#### Layer 13
14x14x512 -- CONV(k=3 p=1) --> 14x14x512 -- MAXPOOL(2) --> 7x7x512

#### Layer 14
7x7x512 -- FLATTEN --> 1835008 -- LINEAR -- RELU --> 4096

#### Layer 15
4096 -- DROPOUT -- LINEAR -- RELU --> 4096

#### Layer 16
4096 -- DROPUT -- LINEAR -- SOFTMAX --> 1000
