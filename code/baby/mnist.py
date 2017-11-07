import os

import numpy
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

import utils.image_util as image_util

tf.logging.set_verbosity("INFO")


def model_fn(features, labels, mode):
    # model
    weight = tf.get_variable(name="weight", shape=[784, 10], dtype=tf.float32)
    bias = tf.get_variable(name="bias", shape=[10], dtype=tf.float32)
    logits = tf.matmul(features['x'], weight) + bias
    y = tf.nn.softmax(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y)
    # loss sub-graph
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    # train sub-graph
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    global_step = tf.train.get_global_step()
    train_op = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # estimator connects graphs
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train_op)


def train(_estimator):
    # input functions
    data_sets = input_data.read_data_sets("resources", one_hot=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": data_sets.train.images},
        data_sets.train.labels,
        batch_size=1,
        num_epochs=10,
        shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": data_sets.validation.images},
        data_sets.validation.labels,
        batch_size=1,
        num_epochs=10,
        shuffle=False)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": data_sets.test.images},
        data_sets.test.labels,
        batch_size=1,
        shuffle=False)

    # train
    _estimator.train(input_fn=train_input_fn, steps=10)
    # eval
    print("eval metrics: {}".format(_estimator.evaluate(input_fn=eval_input_fn)))
    print("test metrics: {}".format(_estimator.evaluate(input_fn=test_input_fn)))


estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir='../checkpoints/mnist')
# train
if not os.path.exists(estimator.model_dir):
    train(estimator)

# predict
image = image_util.read('../resources/8.jpg')
predict_input_fn = tf.estimator.inputs.numpy_input_fn({"x": image}, shuffle=False)
predictions = estimator.predict(predict_input_fn)
for prediction in predictions:
    print("Prediction: {}".format(numpy.argmax(prediction)))
