# ML System Design

## Sidebar
### Kernel trick
TODO

### Kernel approximations
Instead of using the kernel trick, _actually_ estimate the value of a higher dimensional vector phi(x_vec) used by a specific kernel. THe advantage of doing this is that it helps in online learning. SVM using a kernel does not scale well with large datasets. Using linear SVM (or the SGDClassifier) after applying a non-linear transformation to the original vector can be much more scalable. The following kernel approximators are available in scikit-learn.

* Nystroem Method: Uses the RBF kernel for low-rank approximation of kernels.
* RBFSampler: Explicitly constructs a vector that is used implicitly by the RBF kernel.
* Additive chi squared sampler: Explicitly constructs a vector that is used implicitly by the additive chi squared kernel.
* Skewed chi squared sampler: Explicitly constructs a vector that is used implicitly by the skewed chi squared kernel.

### Measuring distance/similarity between instances
There are a number of ways to measure distance between two vectors like the euclidean distance or L2 norm, manhattan distance or L1 norm, etc. On the other hand, the plain old dot product gives the similarity between two vectors. If the dot product is 0, the vectors are identical and so on. Now, various kernels also give the inner products of two vectors transformed onto higher dimensions. So we can use these kernels as a similarity measure. And the kernel distance can be transformed to give the kernel distance between two vectors. Of course it is not a simple inversion, there are various complicated formulae to do this.

## Preprocessing
### Standardization
Operates at a column level. Calculates the mu and sigma for each column and replaces each value with (x - mu) / sigma. This is what Ng calls "normalization".

### Normalization
Scaling individual samples to have unit norm. Operates on a per row basis. Takes each row as a vector, and returns its unit vector.

### Binarization
Thresholding numerical featues to get boolean values. E.g., takes in a matrix where each element is a number between -5 and 5. If we choose the threshold as 0 (default setting), then this module will return a matrix with all numbers <=0 set to 0 and all numbers > 0 set to 1.

### Label encoding
Convert string labels into numerical labels. E.g., a column like -
['paris',
 'paris',
 'tokyo',
 'amsterdam']
has 3 unique values - 'paris', 'tokyo', and 'amsterdam' and each value is assigned a number. In this case 'amsterdam' --> 0, 'paris' --> 1, 'tokyo' --> 2. Note, the numerical values themselves have no ordering, but scikit-learn seems to do a lexicographic sort before assigning values. Now given a column ['tokyo', 'tokyo', 'paris'] it will output [2, 2, 1].

### One hot encoding
Once categorical features have been encoded as numbers (using Label encoding for example), then one hot encoding can be used to exploed each categorical feature into a set of features - one for each unique value so that ML algos do ont inadvertantly assign meaning to the numerical order of these feature labels. E.g., given a matrix 
[[0, 0, 3], 
 [1, 1, 0], 
 [0, 2, 1], 
 [1, 0, 2]]
one hot encoder will figure out a that the first column has two unique values (0, 1), second column as 3 unique values (0, 1, 2), and the third column has four unique values (0, 1, 2, 3). For each column it will assign an index to each unique value. Now given a vector [0, 1, 1] it will output [1, 0, 0, 1, 0, 0, 1, 0, 0] where [(1, 0), (0, 1, 0), (0, 1, 0, 0)] is the column breakdown.

### Label binarization
Conceptually exactly same as one hot encoder except that label binarizer works on a single column (vector) and one hot encoder works on a matrix. E.g., there is a column with values [1, 2, 6, 4, 2], then label binarizer will figure out the unique values in here (1, 2, 4, 6) and will assign an index to each unique value resulting in 1 --> 0, 2 --> 1, 4 --> 2, 6 --> 3. Now if it is a given a number 6, it will return a vector (0, 0, 0, 1). if it is given a column vector -
[1, 
 6] 
it will return a matrix -
[[1, 0, 0, 0], 
 [0, 0, 0, 1]]

### Imputation of missing values
Replaces all missing values (written as np.nan or the string 'NaN') with the mean, mediam, or mode of the column or the row.

## Dimensionality reduction
### Feature hashing
Uses the hashing trick to reduce the number of dimensions in the input dataset. E.g., in a text classification problem, if each word in my corpus vocabulary is a feature, then I'd have tens of thousands of features, but each individual feature will not have any semantic meaning.

### PCA
Takes a dataset with large number of features and converts it to an equivalent dataset with smaller number of features. The resultant features do not have any semantic meaning. TODO: What is the core concept behind PCA?

### Random projections
A vector with high dimensionality is projected onto a random hyperplane of lower dimensionality. The original distance between any two vectors is preserved in the lower dimensionality hyperplan. There are two types of random projections - Gaussian and Sparse.

### Feature agglomeration
Groups together features that behave similarly using hierchical clustering.

### Variance threshold
Remove all features that have variance below a certain threshold. So more-or-less homogenous features are eliminated.

### Univariate feature selection
On each feature perform a chi-square test or an f-test (for classification problems); or a regression test for regression problems, to weed out features that seem to not contribute to the classification of the output variable.

### Recursive feature elimination
Given an estimator that assigns weights to features (like logistic regression), RFE selects features by recusrively considering smaller and smaller sets of features. First the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set of features. This is done recursively on the pruned set until only the desired number of features remain.

## Estimator Parameters
Conceptually, there are two types of paramters that we need to figure out for any ML algo or Estimator according to scikit-learn terminology -

* Classification params
* Training hyper params

Take Logistic Regression as an example, the main learning algo learns the weights vector theta. However, we have to specify the regularization parameter lambda_ and if using gradient descent, then the learning rate alpha to the learning algo. In this case, theta vector are the classification params; lambda and alpha are the hyper params.
While any estimator does a good job of learning the classification params based on the training data, the training hyper params have to be manually set.

## Training set, Validation set, and Test set
An estimator will do pretty well when measured on the data it was trained with. It is important to split the data into training set and test set, so the algorithm can be trained on one set of data and tested on another. However, usually, upon seeing the first test results, we usually fiddle with the process, either changing the training hyper params, or some other aspect of the algo. And then repeat the training and testing until we are satisfied. By now, the test data has had an influence on the estimator params. As such, measuring its performance on the test set is also likely to be optimistic. To prevent this, we usually split the data into 3 parts -

* Training set - usually 60% of the data
* Validation set - usually 20% of the data
* Test set - usually 20% of the data

Now, the training and validation sets are used to fiddle around with the estimator params until we are satisfied. The test set is used to provide a sense of how the estimator will preform in the wild. We should be careful not to make any changes to the algo based on its performance on the test set. This will make the test set performance more optimistic and less realistic.

## Cross-Validation
While the train/validation/test split is good when we have a lot of data to spare, it does "waste" 20% of data that is in the test set as it is never used for training. To conserve training data the cross-validation approach is followed. Here there is no reserved test set. Instead the data is split into _k_ partitions. Then _k-1_ partitions are taken as the training set and the remaining partition is taken as the test set. The estimator is trained on the training set and measured on the test set. This is repeated untill all the _k_ parititions have been the test set. The estimator performance is averaged over the _k_ runs. This means that after _k_ runs, we can fiddle with the algo and do another set of _k_ runs and get the average performance without fear of getting too optimistic a measure. However, it is recommended that we still reserve a test set that is never seen until the final estimation of performance.

Most scikit-learn modules accept a "cv" parameter. This can either be an integer or a cross-validation iterator. If an integer is provided, the data is split into 5 different sets or folds. Then one fold is set aside as the test fold, and the estimator is fit to the remaining 4 folds. Its score is calculated on the test fold. This is repeated 5 times with each fold being set aside as the test fold in turn. This gives 5 different scores. Averaging these scores gives a more realistic estimate of the score on new data.

While modules like GridSearchCV will accept a cv parameter and give scores for each fold, it is possible to use the cross\_val\_score module to calculate the cross-validation scores for some estimator set with some specific hyper params. Like any other scikit-learn module, cross\_val\_score also accepts a "scoring" param allowing us to choose the scoring metric.

When I want to know the predicted value for a particular sample data, I'd want to train the model without that particular sample and then get its prediction. cross\_val\_predict makes it easy to do this. It will predict all the samples in the data when that particular sample was not part of the training set.

Various strategies can be used to split the data into k-folds. The simplest is the KFold strategy which serially splits the data. This is the default for regression estimators used by modules that accept the "cv" param or cross_validation modules. StratifiedKFold ensures that each fold approximately represents the target class percentage. This is the default for classification estimators. Another good alternative is the ShuffleSplit iterator. This shuffles the rows first and then creates a user-specified number of train/test folds. StratifiedShuffleSplit shuffles the rows, but still creates train/test split that are representative of the target labels. Additionally both KFold and StratifiedKFold have a "shuffle" parameter that will shuffle the data before folding. This defaults to False. But these can be used instead of ShuffleSplit iterators.

See cross_val.py for examples.
 
## Performance Measurement
Various modules in scikit-learn accept a "scoring" parameter that tells that module which scoring metric to use. For regression the default is coefficient of determination R^2 metric. For classification the default is the accuracy metric. According to Ng, the following should be used -

* For regression use the [mean squared error](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error)
* For classification use any of the following depending on skewness of data
    - [accuracy](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score)
    - [precision](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score)
    - [recall](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score)
    - [F1 score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score)
    - [log-loss](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss)
    
The string names of various metrics are listed [here](http://scikit-learn.org/stable/modules/model_evaluation.html)

See metrics.py for examples.

### Precision and Recall
Skewed data is where majority of the data has the same classification. Typically in Bayesian type problems, e.g., cancer screening tests, most subjects will test negative for cancer. In this case, having a degenerate classifier that always predicts "negative", will have a very high accuracy. But that is misleading. In such cases it is useful to use precision, recall, and F1 score as measures of performance instead of accuracy.

> _Precision_: Of all the estimator's positive predictions, how many did it get right.

> _Recall_: Of all the positive data, how many did the estimator get right.

If I want the estimator to be conservative and not miss any positive case, I want it to have high recall. In the degenerate case of the above example, if my estimator always predicts "positive", it will have 100% recall, but very poor precision. To get a sense of both precision and recall, their harmonic mean called its F1 score is taken.

> _F1 Score_ = (2 * P * R) / (P + R)

It is tempting to think that precision is same as accuracy, but it isn't. Accuracy takes into account both the negative and positive classifications. Precision and recall, on the other hand only take the positive classifications into account. In that sense, accuracy is more holistic (though harsh) measure of performance. 

For multi-class data, the presision, recall, and F1 score are calculated for individual classes (one-vs-all) and then averaged over. There are different ways of averaging them as provided by scikit-learn.metrics module. A simple average would work for balanced data set where all classes appear more-or-less equal number of times. Otherwise, the weighted method makes most sense, where a weighted average with each class given a weight according to its frequency of appearance.

### Confusion Matrix
Matrix of actual vs. predicted values. There is a row/col for each label. The diagonal represents the correct classifications. Every other cell represents some sort of wrong classification.

TODO: Picture of confusion matrix

> Precision =  True Positives / Predicted Positives = True Positives / (True Positives + False Positives)

> Recall = True Positives / Actual Positives = True Positives / (True Positives + False Negatives)

See metric.py for an example of printing the confusion matrix.

## Bias and Variance
ML algos usually suffer from one of _bias_ (underfit) or _variance_ (overfit). When measuring performance, we are usually trying to figure out which, and then making the approriate changes to the algo.

TODO: Picture of training score vs. validation score

> _Bias_ is when the model is way too simple for the data and it underfits the data

> _Variance_ is when the model is way too complex for the data and it overfits the data

Comparing the training error with the validation (or test) error gives a good idea of whether the estimator has high bias or high variance. In general -

> training error << test error => high variance (overfit)

> training error = test error and they are both high => high bias (underfit)


## Learning Curves
When analyzing a particular estimator parameter - say lambda_ - it is useful to draw the graph of estimator scores on training and validation data sets on the y-axis, and different parameter values on the x-axis. Generally these are called validation curves. A more common graph is the esitmator train and validation scores vs. different training data sizes. This is called the learning curve. It tells us whether or not getting additional data will help.

See learning_curves.py for examples.

## Workflow
Typical steps to follow when creating a new model. Somewhere in the workflow it is useful to use the [Dummy estimators](http://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators) to explore the data.

### Data Exploration
TODO

### Feature Selection and Extraction
TODO

### Choosing Scoring Method
If this is a regression problem, choose the mean squared error method. If this is a classification problem check the skewness of data. If data is not too skewed, then choose the accuracy metric. If it is skewed and you need to balance precision and recall, choose F1.

### Choosing Data Split
If you have enough data, split the data into train and test set. Set aside the test set for the end. Use the train set for the remaining steps.

### Implement and Measure Estimator
Start with a simple algorithm that you can implement quickly and train it with the training data and measure its average performance on the cross-validation sets.

### Plot Learning Curves
Decide whether or not more data is likely to help. Get a sense of bais/variance of the estimator.

### Manual Error Analysis
Manually examine samples from the validation set that your algo made errors on. See if you can spot any systemic trend on the errorneous classifications. Make changes to the estimtor params. For example, for a text classifier, it could be to choose between stemming or no stemming, using stopwords or not using stopwords, etc. Draw validation curves for any parameter that is being varied.

### Next Steps
If the scores are still not high enough, then either my model is too simple (it has high bias) and I need to make it more complex; or my model is too complex (it has high variance) and I need to make it simpler. One of the following steps will help -

#### Make the model more complex to fix high bias
* Increase number of features by say adding polynomial features
* Decrease lambda_ 
* Increase number of nodes in the hidden layer

#### Make the model simpler to fix high variance
* Get more training examples
* Reduce number of features
* Increase lambda_

## Regularization
#### L2: Minimizes the weights whose change does not contribute much to the cost.

#### L1: Results in sparse weight tensors.

#### Dataset Augmentation
For images the following can be done, but take care the original class of the image does not change:

  * Horizontal swap
  * Rotate
  * Crop
  * Translate a few pixels

#### Add Noise

  * Inject random noise to inputs
  * Inject random noise to hidden layers
  * Perturb the weights everytime a new sample is presented

#### Label Smoothing
In softmax training, replace the labels $$0, 1$$ with $$\frac {\epsilon}{k-1}$$ and $$1 - \epsilon$$. Then train the classifier with these smoothed out labels.

#### Multitask Learning
In a typical multi-headed network, where this a single output, but we are trying to predict multiple things, having a stack of common layers, which then branch out results in a shared representation shared by the tasks. This acts as a regularizer because it forces the weights to generalize better.

#### Early Stopping
Typically validation learning curves are U-shaped, where the cost starts increasing as the model starts overfitting. In this case, we don't take the latest learned params but rather the params that were learned when the vlaidation loss was at its lowest.

#### Parameter Tying and Parameter Sharing
Kernel sharing. Why is this a regularization mechanism?

#### Sparse Representation

#### Bagging and Ensemble

#### Dropout

#### Adversarial Training

#### Tangent Distance, Tangent Prop, Manifold Tangent Classifier


















