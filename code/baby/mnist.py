import os

import numpy
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

from utils import image_util

tf.logging.set_verbosity("DEBUG")

TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
EPOCHS_SIZE = 10
BATCH_SIZE = 100

if __name__ == '__main__':
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=64, input_dim=784))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dense(units=10))
    model.add(tf.keras.layers.Activation("softmax"))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
        metrics=['accuracy'])
    estimator = tf.keras.estimator.model_to_estimator(model, model_dir='/project/code/checkpoints/mnist')

    if not os.path.exists(estimator.model_dir + "/checkpoint"):
        # input functions
        data_sets = input_data.read_data_sets("../resources", one_hot=True)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'dense_1_input': data_sets.train.images},
            numpy.asarray(data_sets.train.labels, dtype=numpy.float32),
            num_epochs=EPOCHS_SIZE,
            batch_size=BATCH_SIZE,
            shuffle=True)
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'dense_1_input': data_sets.validation.images},
            numpy.asarray(data_sets.validation.labels, dtype=numpy.float32),
            batch_size=BATCH_SIZE,
            shuffle=False)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            {'dense_1_input': data_sets.test.images},
            numpy.asarray(data_sets.test.labels, dtype=numpy.float32),
            batch_size=BATCH_SIZE,
            shuffle=False)

        # train
        estimator.train(input_fn=train_input_fn, steps=TRAIN_SIZE/BATCH_SIZE*EPOCHS_SIZE)
        # eval
        evaluate = estimator.evaluate(input_fn=eval_input_fn)
        print("eval metrics: {}".format(evaluate))
        test = estimator.evaluate(input_fn=test_input_fn)
        print("test metrics: {}".format(test))

    # predict
    image = image_util.read('../resources/8.jpg')
    predict_input_fn = tf.estimator.inputs.numpy_input_fn({"dense_1_input": image}, shuffle=False)
    predictions = estimator.predict(predict_input_fn)
    for prediction in predictions:
        print("Prediction: {}".format(numpy.argmax(prediction['activation_2'])))
