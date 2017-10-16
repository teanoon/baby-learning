import tensorflow as tf
import numpy as np


# args names are restricted
def model_fn(features, labels, mode):
    # model
    _weight = tf.get_variable(name="weight", shape=[1], dtype=tf.float32)
    _bias = tf.get_variable(name="bias", shape=[1], dtype=tf.float32)
    linear_model = features['x'] * _weight + _bias
    # loss sub-graph
    loss = tf.reduce_sum(tf.square(linear_model - labels))
    # train sub-graph
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    global_step = tf.train.get_global_step()
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # estimator connects graphs
    return tf.estimator.EstimatorSpec(mode=mode, predictions=linear_model, loss=loss, train_op=train)


estimator = tf.estimator.Estimator(model_fn=model_fn)

# data sets
x_train = np.array([1., 2., 3., 4.], dtype=np.float32)
y_train = np.array([5., 8., 11., 14.], dtype=np.float32)
x_eval = np.array([5., 6., 7., 8.], dtype=np.float32)
y_eval = np.array([17., 20., 23., 26.], dtype=np.float32)

# input functions
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train
estimator.train(input_fn=input_fn, steps=1000)
# eval
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r" % train_metrics)
print("eval metrics: %r" % eval_metrics)
