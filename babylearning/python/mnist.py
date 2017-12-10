from os.path import join

import numpy
import tensorflow as tf
from hyperopt import hp, fmin, tpe, STATUS_OK
from tensorflow.examples.tutorials.mnist import input_data

from babylearning.python.definitions import CKPT_DIR, RES_DIR
from babylearning.python.model import linear
from babylearning.python.utils import image
from babylearning.python.utils.hooks import IteratorInitializerHook

tf.logging.set_verbosity("WARNING")

TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
SHUFFLE_BUFFER_SIZE = 10000

MNIST_DIR = join(CKPT_DIR, 'mnist4')

EPOCHS_SIZE = 5
BATCH_SIZE = 100


def input_fn(features, labels, batch_size=BATCH_SIZE, is_train=True):
    iterator_initializer_hook = IteratorInitializerHook()
    _features = numpy.asarray(features, dtype=numpy.float32)
    _labels = numpy.asarray(labels, dtype=numpy.float32)

    def _input_fn():
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


def build_objective(epochs=1, is_inference=False):
    _data_sets = input_data.read_data_sets(RES_DIR, one_hot=True)

    def _objective(params):
        objective_params = {**params}
        batch_size = int(params.pop('batch_size'))

        _train_input_fn, _train_iterator_hook = input_fn(
            _data_sets.train.images,
            _data_sets.train.labels,
            batch_size=batch_size)
        _eval_input_fn, _eval_iterator_hook = input_fn(
            _data_sets.validation.images,
            _data_sets.validation.labels,
            batch_size=batch_size,
            is_train=False)

        _estimator = linear.keras(784, **params)

        _result = {}
        for _ in range(epochs):
            _estimator.train(
                _train_input_fn,
                hooks=[_train_iterator_hook],
                steps=TRAIN_SIZE / batch_size)

            _result = _estimator.evaluate(
                _eval_input_fn,
                hooks=[_eval_iterator_hook])

        if is_inference:
            _test_input_fn, _test_iterator_hook = input_fn(
                _data_sets.test.images,
                _data_sets.test.labels,
                batch_size=batch_size,
                is_train=False)
            _test = _estimator.evaluate(
                input_fn=_test_input_fn, hooks=[_test_iterator_hook])
            print("test metrics: {}".format(_test))
            return _estimator
        else:
            _result = {**_result, **objective_params}
            print("objective metrics: {}".format(_result))
            _result['status'] = STATUS_OK
            return _result

    return _objective


if __name__ == '__main__':
    objective = build_objective()
    space = {
        'batch_size': hp.uniform('batch_size', BATCH_SIZE - 10, BATCH_SIZE + 10),
        'learning_rate': hp.uniform('learning_rate', 1e-3, 2e-2),
        'momentum': hp.uniform('momentum', 0.45, 1.35),
        'decay': hp.uniform('decay', 1e-7, 2e-6)
    }
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=3)
    print(best)
    inference = build_objective(EPOCHS_SIZE, is_inference=True)
    estimator = inference(best)

    # predict
    image = image.read(RES_DIR + '/8.jpg')
    image = numpy.reshape(image, [1, 784])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn({'x_input': image}, shuffle=False)
    predictions = estimator.predict(predict_input_fn)
    for prediction in predictions:
        print("Prediction: {}".format(numpy.argmax(prediction['output'])))
