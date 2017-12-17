"""
Train entrypoint
"""
import os

import numpy
import tensorflow as tf
from hyperopt import hp, fmin, tpe, STATUS_OK
from tensorflow.examples.tutorials.mnist import input_data

from trainer.model import linear
from trainer.utils.hooks import IteratorInitializerHook


TRAIN_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
SHUFFLE_BUFFER_SIZE = 10000

INPUT_TENSOR_NAME = 'x_input'

BATCH_SIZE = 100
LOG_DIR = 'log'
MODEL_DIR = 'model'


# TODO move to a common util
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

        return {INPUT_TENSOR_NAME: next_features}, next_labels

    return _input_fn, iterator_initializer_hook


def serving_input_receiver_fn():
    """
    An input receiver that expects a list of image array.

    Returns:
        A serving_input_receiver_fn suitable for use in serving.
    """
    # TODO unify INPUT_TENSOR_NAME, feature column definitions
    feature_spec = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, shape=[1, 784])}
    return tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)()


def build_objective(data_sets, epochs=1):
    def _objective(params, model_dir=None):
        objective_params = dict(params.items())
        batch_size = int(params.pop('batch_size'))

        _train_input_fn, _train_iterator_hook = input_fn(
            data_sets.train.images,
            data_sets.train.labels,
            batch_size=batch_size)
        _eval_input_fn, _eval_iterator_hook = input_fn(
            data_sets.validation.images,
            data_sets.validation.labels,
            batch_size=batch_size,
            is_train=False)

        _estimator = linear.keras(784, model_dir=model_dir, **params)

        _result = {}
        for _ in range(epochs):
            _estimator.train(
                _train_input_fn,
                hooks=[_train_iterator_hook],
                steps=TRAIN_SIZE / batch_size)

            _result = _estimator.evaluate(
                _eval_input_fn,
                hooks=[_eval_iterator_hook])

        _result = dict(_result.items() + objective_params.items())
        tf.logging.info("objective metrics: {}".format(_result))
        _result['status'] = STATUS_OK

        if model_dir is not None:
            _estimator.export_savedmodel(model_dir, serving_input_receiver_fn)

        return _result

    return _objective


def main(argv):
    data_sets = input_data.read_data_sets(argv.input_dir, one_hot=True)
    objective = build_objective(data_sets)
    space = {
        'batch_size': hp.uniform('batch_size', BATCH_SIZE - 10, BATCH_SIZE + 10),
        'learning_rate': hp.uniform('learning_rate', 1e-3, 2e-2)
    }
    best = fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=argv.max_evals)
    tf.logging.info('Best hyperparameters are {}'.format(best))

    train = build_objective(data_sets, argv.epochs)
    if not argv.job_dir.startswith('gs') and not os.path.exists(argv.job_dir):
        os.mkdir(argv.job_dir)
    result = train(best, model_dir=argv.job_dir)
    tf.logging.info(result)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_integer('epochs', 2, 'Number of steps to run trainer.')
    # gcloud uses dash style but python needs underscore style
    flags.DEFINE_integer('max-evals', 1, 'Number of steps to run trainer.', short_name='max_evals')
    flags.DEFINE_string('input-dir', '../data', 'Directory to read the training data.', short_name='input_dir')
    flags.DEFINE_string('job-dir', '../job', 'Directory to put the model into.', short_name='job_dir')

    # TODO add a common flag
    tf.logging.set_verbosity("INFO")

    main(flags.FLAGS)
