import numpy
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn.python.learn.datasets as datasets
import tensorflow.contrib.learn.python.learn.estimators as estimators

# load datasets
training_set = datasets.base.load_csv_with_header(
    filename="resources/linear_training.csv",
    target_dtype=numpy.int,
    features_dtype=numpy.int)
test_set = datasets.base.load_csv_with_header(
    filename="resources/linear_test.csv",
    target_dtype=numpy.int,
    features_dtype=numpy.int)

# build linear model
print('building...')
x_column = [layers.real_valued_column('')]
regressor = estimators.LinearRegressor(x_column)

# train
print('training...')
regressor.fit(
    x=training_set.data,
    y=training_set.target,
    steps=200)

# evaluate
print('evaluating...')
score = regressor.evaluate(
    x=test_set.data,
    y=test_set.target)
print("Results: {}".format(str(score)))

# predict
print("predicating")
new_samples = numpy.array([[10], [11], [12]])
predicts = list(regressor.predict(new_samples, as_iterable=True))
print("Predicts: {}".format(str(predicts)))
