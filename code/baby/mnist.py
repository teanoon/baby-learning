import os

import numpy
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import model.linear_model
from utils import image_util
from utils import hooks

tf.logging.set_verbosity("DEBUG")

TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
EPOCHS_SIZE = 10
BATCH_SIZE = 100

if __name__ == '__main__':
    estimator = tf.estimator.Estimator(
        model_fn=model.linear_model.linear,
        model_dir='../checkpoints/mnist')

    if not os.path.exists(estimator.model_dir + "/checkpoint"):
        # input functions
        data_sets = input_data.read_data_sets("../resources", one_hot=True)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': data_sets.train.images},
            data_sets.train.labels,
            num_epochs=EPOCHS_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': data_sets.validation.images},
            data_sets.validation.labels,
            batch_size=BATCH_SIZE,
            shuffle=False)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'x': data_sets.test.images},
            data_sets.test.labels,
            batch_size=BATCH_SIZE,
            shuffle=False)

        # train
        estimator.train(input_fn=train_input_fn)
        # eval
        evaluate = estimator.evaluate(input_fn=eval_input_fn)
        print("eval metrics: {}".format(evaluate))
        test = estimator.evaluate(input_fn=test_input_fn)
        print("test metrics: {}".format(test))

    # predict
    image = image_util.read('../resources/8.jpg')
    predict_input_fn = tf.estimator.inputs.numpy_input_fn({"x": image}, shuffle=False)
    predictions = estimator.predict(predict_input_fn)
    for prediction in predictions:
        print("Prediction: {}".format(numpy.argmax(prediction)))
