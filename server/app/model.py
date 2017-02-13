import math
from functools import reduce

import tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.training import learning_rate_decay

D_TYPE = dtypes.float32


# helpers
def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensorflow.summary.histogram(x.op.name + '/activations', x)
    tensorflow.summary.scalar(x.op.name + '/sparsity', tensorflow.nn.zero_fraction(x))


def create_variable(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tensorflow.device('/cpu:0'):
        var = tensorflow.get_variable(name, shape, initializer=initializer, dtype=D_TYPE)
    return var


def create_variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    #  for different purposes like training, evaluating or predicting
    var = create_variable(
        name,
        shape,
        tensorflow.truncated_normal_initializer(stddev=stddev, dtype=D_TYPE))
    if wd is not None:
        weight_decay = tensorflow.mul(tensorflow.nn.l2_loss(var), wd, name='weight_loss')
        tensorflow.add_to_collection('losses', weight_decay)
    return var


# inference - build model
# layers are consist of convolution layer with ReLu activation and max pooling
# each convolution layer should use different kernel
def common_layer(name, images, weight_shape, stddev, wd, initial_constant, conv_func, linear_func, activation_func):
    with tensorflow.variable_scope(name) as scope:
        weights = create_variable_with_weight_decay(
            'weights', shape=weight_shape, stddev=stddev, wd=wd)
        biases = create_variable(
            'biases', weight_shape[-1:], tensorflow.constant_initializer(initial_constant))
        weighted = conv_func(images, weights)
        pre_activation = linear_func(weighted, biases)
        activation = activation_func(pre_activation, name=scope.name)

    return activation


def conv2d(images, kernel):
    return tensorflow.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')


def pool(name, activation):
    return tensorflow.nn.max_pool(
        activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def normalize(name, pooling):
    return tensorflow.nn.lrn(
        pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def inference(images, num_classes=10):
    # 1st hidden layer
    conv1 = common_layer(
        'convolution-1', images, [5, 5, 1, 32], 5e-2, 0.0, 0.0,
        conv2d, tensorflow.nn.bias_add, tensorflow.nn.relu)
    pool1 = pool('pool-1', conv1)
    norm1 = normalize('norm-1', pool1)

    # 2st hidden layer
    conv2 = common_layer(
        'convolution-2', norm1, [5, 5, 32, 32], 5e-2, 0.0, 0.1,
        conv2d, tensorflow.nn.bias_add, tensorflow.nn.relu)
    # TODO why change order?
    norm2 = normalize('norm-2', conv2)
    pool2 = pool('pool-2', norm2)

    # 3rd hidden dense connected layer
    # Move everything into depth so we can perform a single matrix multiply.
    dim = reduce(lambda x, y: x * y, pool2.get_shape()[1:]).value
    reshape = tensorflow.reshape(pool2, shape=[-1, dim])
    local3_weight_stddev = 1 / math.sqrt(dim)
    local3 = common_layer(
        'local3', reshape, [dim, 384], local3_weight_stddev, 0.004, 0.1,
        tensorflow.matmul, tensorflow.add, tensorflow.nn.relu)

    # 4th hidden dense connected layer
    local4_weight_stddev = 1 / math.sqrt(384)
    local4 = common_layer(
        'local4', local3, [384, 192], local4_weight_stddev, 0.004, 0.1,
        tensorflow.matmul, tensorflow.add, tensorflow.nn.relu)

    # 5th hidden layer but not activated
    return common_layer(
        'softmax_linear', local4, [192, num_classes], 1 / 192, 0.0, 0.0,
        tensorflow.matmul, tensorflow.add, lambda x, name: x)


# calculate the loss
def loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = math_ops.cast(labels, dtypes.int64)
    cross_entropy = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tensorflow.reduce_mean(cross_entropy, name='cross_entropy')

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # see #create_variable_with_weight_decay
    tensorflow.add_to_collection('losses', cross_entropy_mean)
    _loss = tensorflow.add_n(tensorflow.get_collection('losses'), name='total_loss')
    tensorflow.summary.scalar('loss', _loss)

    return _loss


# calculate the evaluation accuracy during training
def validate(logits, labels):
    logits = math_ops.cast(logits, dtypes.float32)
    labels = math_ops.cast(labels, dtypes.int64)
    correct = tensorflow.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    correct = tensorflow.reduce_sum(tensorflow.cast(correct, tensorflow.int32))
    shape = tensorflow.shape(logits)
    rate = correct / shape[0] * 1.0
    tensorflow.summary.scalar('accuracy', rate)

    return rate


# setup gradient decent optimizer
def training(_loss, global_step,
             initial_learning_rate=0.1,
             learning_rate_decay_factor=0.1,
             num_steps_per_decay=110):
    learning_rate = learning_rate_decay.exponential_decay(
        initial_learning_rate, global_step, num_steps_per_decay, learning_rate_decay_factor)
    tensorflow.summary.scalar('learning_rate', learning_rate)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    return optimizer.minimize(_loss, global_step=global_step)


# calc softmax for predictions
def softmax(logits):
    return tensorflow.nn.softmax(logits)
