import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import SGD


def simple(features, labels, mode):
    # model
    weight = tf.get_variable(
        name="weight", shape=[784, 10],
        dtype=tf.float32, initializer=tf.zeros_initializer)
    bias = tf.get_variable(
        name="bias", shape=[10],
        dtype=tf.float32, initializer=tf.zeros_initializer)
    logits = tf.matmul(features['x_input'], weight) + bias
    y = tf.nn.softmax(logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=y)
    # loss sub-graph
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')

    # train sub-graph
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    global_step = tf.train.get_global_step()
    train_op = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))

    # estimator connects graphs
    return tf.estimator.EstimatorSpec(mode=mode, predictions=y, loss=loss, train_op=train_op)


def keras(input_dim, model_dir=None):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim, name='x'))
    model.add(Dense(10, activation='softmax', name='output'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True),
        metrics=['accuracy'])
    return tf.keras.estimator.model_to_estimator(model, model_dir=model_dir)
