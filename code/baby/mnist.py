import os

import numpy
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import utils.hooks as hooks
import utils.image_util as image_util
from definitions import MNIST_DIR
from definitions import RES_DIR

tf.logging.set_verbosity("DEBUG")

TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
EPOCHS_SIZE = 10
BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 10000

RETRAIN = True


def input_fn(features, labels, batch_size=BATCH_SIZE, is_train=True, shuffle=True):
    iterator_initializer_hook = hooks.IteratorInitializerHook()

    def _input_fn():
        features_placeholder = tf.placeholder(features.dtype, features.shape)
        labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder)).batch(batch_size)
        if shuffle:
            dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
        if is_train:
            dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        next_features, next_labels = iterator.get_next()

        iterator_initializer_hook.iterator_initializer_func = \
            lambda session: session.run(iterator.initializer, feed_dict={
                features_placeholder: features,
                labels_placeholder: labels
            })

        return {'x_input': next_features}, next_labels

    return _input_fn, iterator_initializer_hook


if __name__ == '__main__':
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=784, name='x'))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name='output'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
        metrics=['accuracy'])
    estimator = tf.keras.estimator.model_to_estimator(model, model_dir=MNIST_DIR)

    if RETRAIN:
        data_sets = input_data.read_data_sets(RES_DIR, one_hot=True)

        # train
        train_input_fn, train_iterator_hook = input_fn(
            numpy.asarray(data_sets.train.images, dtype=numpy.float32),
            numpy.asarray(data_sets.train.labels, dtype=numpy.float32))
        estimator.train(
            input_fn=train_input_fn,
            hooks=[train_iterator_hook],
            steps=TRAIN_SIZE/BATCH_SIZE*EPOCHS_SIZE)

        # eval
        validation_input_fn, validation_iterator_hook = input_fn(
            numpy.asarray(data_sets.validation.images, dtype=numpy.float32),
            numpy.asarray(data_sets.validation.labels, dtype=numpy.float32),
            is_train=False)
        evaluate = estimator.evaluate(
            input_fn=validation_input_fn,
            hooks=[validation_iterator_hook])
        print("eval metrics: {}".format(evaluate))

        # test
        test_input_fn, test_iterator_hook = input_fn(
            numpy.asarray(data_sets.test.images, dtype=numpy.float32),
            numpy.asarray(data_sets.test.labels, dtype=numpy.float32),
            is_train=False)
        test = estimator.evaluate(
            input_fn=test_input_fn,
            hooks=[test_iterator_hook])
        print("test metrics: {}".format(test))

    # predict
    image = image_util.read('../resources/8.jpg')
    predict_input_fn = tf.estimator.inputs.numpy_input_fn({'x_input': image}, shuffle=False)
    predictions = estimator.predict(predict_input_fn)
    for prediction in predictions:
        print("Prediction: {}".format(numpy.argmax(prediction['output'])))
