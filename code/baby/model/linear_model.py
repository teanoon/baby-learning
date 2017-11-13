import tensorflow as tf


def linear(features, labels, mode):
    # model
    weight = tf.get_variable(
        name="weight", shape=[784, 10],
        dtype=tf.float32, initializer=tf.zeros_initializer)
    bias = tf.get_variable(
        name="bias", shape=[10],
        dtype=tf.float32, initializer=tf.zeros_initializer)
    logits = tf.matmul(features['x'], weight) + bias
    y = tf.nn.softmax(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y)
    # loss sub-graph
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.losses.compute_weighted_loss(loss, reduction=tf.losses.Reduction.SUM)
    # train sub-graph
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    global_step = tf.train.get_global_step()
    train_op = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # estimator connects graphs
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train_op)
