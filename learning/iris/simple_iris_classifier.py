import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn.python.learn.datasets as datasets
import tensorflow.contrib.learn.python.learn.estimators as estimators
import numpy as np

# Load datasets.
training_set = datasets.base.load_csv_with_header(
    filename="resources/iris_training.csv",
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = datasets.base.load_csv_with_header(
    filename="resources/iris_test.csv",
    target_dtype=np.int,
    features_dtype=np.float32)

# Specify that all features have real-value data
# feature dimension is defined in csv file.
feature_columns = [layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = estimators.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 20, 10],
    n_classes=3,
    model_dir="/tmp/iris_model")

# Fit model.
classifier.fit(
    x=training_set.data,
    y=training_set.target,
    steps=2000)

# Evaluate accuracy.
score = classifier.evaluate(
    x=test_set.data,
    y=test_set.target)
print('Accuracy: {0:f}'.format(score["accuracy"]))

# Classify two new flower samples.
new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print('Predictions: {}'.format(str(y)))
