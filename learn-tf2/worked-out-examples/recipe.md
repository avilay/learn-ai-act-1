# Recipe
My default style choice is the Keras Functional API because it offers a good mix of flexibility and convenience. Only use the gradient tape style if functional APIs cannot meet the requirement. The following steps take me from starting out with data to a fully trained model.

### 1. Data Exploration
This will vary on a case-by-case basis. I need to understand what sort of feature engineering will be required, what kind of data pipeline will be needed and how will I integrate this with `tf.data.Dataset` objects.

### 2. Split Data
Here I'll integrate with `tf.data.Dataset` and figure out how to split the data. At the end of this phase I should have at least three datasets - train_dsname, val_dsname, test_dsname. In this phase I need to make the following hyperparameter choices:
  1. What validation strategy am I using? e.g., fixed validation set, cross-validation, etc.
  2. What percent of data will I set aside for validation?

### 3. Data Pipeline
Implement the design I came up with in the Data Exploration phase. The code will look something like this -
```python
def func1(elem):
    pass

def func2(elem):
    pass

train_ds = train_dsname.map(func1)
train_ds = train_ds.map(func2)

val_ds = val_dsname.map(func1)
val_ds = val_ds.map(func2)

test_ds = test_dsname.map(func1)
test_ds = test_ds.map(func2)

for x, y in train_ds.take(5):
    examine(x, y)
```
I'll not add the shuffling and batching parts of the pipeline just yet.

### 3. Build Model
Here I'll design the model using Tensorflow Hub and Keras functional APIs. Here I need to make hyperparameter choices related to the architecture.
  1. What approach to use?
  2. Are there existing pre-trained models that can be used for this?
  3. How many layers, what kind of layers, what connection topology?
  4. What kind of architecture regularization should I use? Should I use batch norm? If so where? Should I use dropouts? If so where and what percent?

This is what the code will roughly look like:
```python
input_ = layers.Input(shape=shape)
x = layers.SomeLayer()(input_)
:
output = layers.SomeLayer()(x)
model = keras.Model(input_, output)
keras.utils.plot_model(model, show_shapes=True)
```

Before I end this phase, I need to try out the mode on a single batch and see if it works.
```python
sample_batch = None
for batch_x, batch_y in train_ds.batch(8).take(1):
    sample_batch = batch_x
    
sample_pred = model(sample_batch)
sample_pred.shape
```

### 4. Train Model
Here I need to make the following hyperparameter choices:
  1. Which optimizer to use?
  2. How to parameterize the optimzer? e.g., what kind of learning rate, momentum rate, etc.?
  3. Which loss function to use?
  4. Which evaluation metrics to use?
  5. What is the shuffle buffer size?
  6. What is the batch size?
  7. How many epochs to run?
  
Additionally, I also need to make design choices w.r.t callbacks. At the very least I'll need to use the TensorBoard callback.  
  
```python
model.compile(
    optimizer=optim, 
    loss=loss_fn,
    metrics=[met1, met2])

tb = keras.callbacks.TensorBoard(log_dir="./dsname.tb", histogram_freq=1, update_freq="epoch")

model.fit(
    train_ds.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE),
    validation_data=val_ds.batch(BATCH_SIZE),
    epochs=epochs,
    callbacks=[tb]
)
```

### 5. Fine Tune
Initial results will be bad. Have the model predict the validation set and examine all the instances that it got wrong. From that try to get an insight on how to improve the model. The `predict` API also works on batches. If I want to predict a single instance, I'll have to create a batch of one.
```python
preds = model.predict(val_ds.batch(SOME_VERY_LARGE_NUM))
```

### 5. Evaluate Model
Here I just run the model on the test data and gather the test metrics.
```python
test_stats = model.evaluate(test_ds.batch(SOME_VERY_LARGE_NUM))
```
