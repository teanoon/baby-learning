from os.path import join

import numpy
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from babylearning.python.definitions import CKPT_DIR
from babylearning.python.definitions import RES_DIR
from babylearning.python.model import linear
from babylearning.python.utils import image
from babylearning.python.utils.hooks import IteratorInitializerHook

tf.logging.set_verbosity("DEBUG")

TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
EPOCHS_SIZE = 5
BATCH_SIZE = 100
SHUFFLE_BUFFER_SIZE = 10000

RETRAIN = True

MNIST_DIR = join(CKPT_DIR, 'mnist4')


def input_fn(features, labels, batch_size=BATCH_SIZE, is_train=True):
    iterator_initializer_hook = IteratorInitializerHook()

    def _input_fn():
        _features, _labels = features, labels
        features_placeholder = tf.placeholder(_features.dtype, _features.shape)
        labels_placeholder = tf.placeholder(_labels.dtype, _labels.shape)
        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))\
            .batch(batch_size)
        if is_train:
            dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE).repeat()
        iterator = dataset.make_initializable_iterator()
        next_features, next_labels = iterator.get_next()

        iterator_initializer_hook.iterator_initializer_func = \
            lambda session: session.run(iterator.initializer, feed_dict={
                features_placeholder: _features,
                labels_placeholder: _labels
            })

        return {'x_input': next_features}, next_labels

    return _input_fn, iterator_initializer_hook


if __name__ == '__main__':
    estimator = linear.keras(784, model_dir=MNIST_DIR)
    data_sets = input_data.read_data_sets(RES_DIR, one_hot=True)

    # train
    train_input_fn, train_iterator_hook = input_fn(
        numpy.asarray(data_sets.train.images, dtype=numpy.float32),
        numpy.asarray(data_sets.train.labels, dtype=numpy.float32))

    # eval
    eval_input_fn, eval_iterator_hook = input_fn(
        numpy.asarray(data_sets.validation.images, dtype=numpy.float32),
        numpy.asarray(data_sets.validation.labels, dtype=numpy.float32),
        is_train=False)

    # test
    test_input_fn, test_iterator_hook = input_fn(
        numpy.asarray(data_sets.test.images, dtype=numpy.float32),
        numpy.asarray(data_sets.test.labels, dtype=numpy.float32),
        is_train=False)

    for _ in range(EPOCHS_SIZE):
        step_per_epoch = TRAIN_SIZE / BATCH_SIZE
        estimator.train(
            train_input_fn,
            hooks=[train_iterator_hook],
            steps=step_per_epoch)

        estimator.evaluate(
            eval_input_fn,
            hooks=[eval_iterator_hook])

    test = estimator.evaluate(
        input_fn=test_input_fn,
        hooks=[test_iterator_hook])
    print("test metrics: {}".format(test))

    # predict
    image = image.read(RES_DIR + '/8.jpg')
    image = numpy.reshape(image, [1, 784])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn({'x_input': image}, shuffle=False)
    predictions = estimator.predict(predict_input_fn)
    for prediction in predictions:
        print("Prediction: {}".format(numpy.argmax(prediction['output'])))
