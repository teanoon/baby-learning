import tensorflow
from numpy import array
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes
from tensorflow.python.training import learning_rate_decay
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
BATCH_SIZE = 100
# TODO dynamically read from mnist data set
IMAGE_SIZE = 28 * 28
MAX_STEPS = 2000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
D_TYPE = dtypes.float16

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


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
    tensor_name = tensorflow.re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensorflow.summary.histogram(tensor_name + '/activations', x)
    tensorflow.summary.scalar(tensor_name + '/sparsity', tensorflow.nn.zero_fraction(x))


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
    var = create_variable(
        name,
        shape,
        tensorflow.truncated_normal_initializer(stddev=stddev, dtype=D_TYPE))
    if wd is not None:
        weight_decay = tensorflow.mul(tensorflow.nn.l2_loss(var), wd, name='weight_loss')
        tensorflow.add_to_collection('losses', weight_decay)
    return var


# input
# read data set and divide them into train/test/eval 3 groups
# train group should go through data augmentation
# TODO w/o streaming
def input_with_type(data_set_type, file_path='resources'):
    """Get mnist data use tensorflow mnist
    """
    if data_set_type not in ['train', 'test', 'validation']:
        raise ValueError('Failed to find proper type in: train, test, validation')
    data = input_data.read_data_sets(file_path)
    return getattr(data, data_set_type)


# inference - build model
# layers are consist of convolution layer with ReLu activation and max pooling
# each convolution layer should use different kernel
def common_layer(name, images, weight_shape, conv_func, linear_func, activation_func):
    with tensorflow.variable_scope(name) as scope:
        weights = create_variable_with_weight_decay(
            'weights', shape=weight_shape, stddev=0.05, wd=0.0)
        biases = create_variable(
            'biases', weight_shape[-1:], tensorflow.constant_initializer(0.0))
        weighted = conv_func(images, weights)
        pre_activation = linear_func(weighted, biases)
        activation = activation_func(pre_activation, name=scope.name)
        # activation_summary(activation)

    return activation


def conv2d(images, kernel):
    return tensorflow.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')


def pool(name, activation):
    return tensorflow.nn.max_pool(
        activation, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def normalize(name, pooling):
    return tensorflow.nn.lrn(
        pooling, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def inference(images):
    # 1st hidden layer
    conv1 = common_layer('convolution-1', images, [5, 5, 1, 32], conv2d, tensorflow.nn.bias_add, tensorflow.nn.relu)
    pool1 = pool('pool-1', conv1)
    norm1 = normalize('norm-1', pool1)

    # 2st hidden layer
    conv2 = common_layer('convolution-2', norm1, [5, 5, 32, 32], conv2d, tensorflow.nn.bias_add, tensorflow.nn.relu)
    # TODO why change order?
    norm2 = normalize('norm-2', conv2)
    pool2 = pool('pool-2', norm2)

    # TODO why these two layers?
    # 3rd hidden layer
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tensorflow.reshape(pool2, [pool2.get_shape()[0].value, -1])
    dim = reshape.get_shape()[1].value
    local3 = common_layer('local3', reshape, [dim, 384], tensorflow.matmul, tensorflow.add, tensorflow.nn.relu)

    # 4th hidden layer
    local4 = common_layer('local4', local3, [384, 192], tensorflow.matmul, tensorflow.add, tensorflow.nn.relu)

    # 5th hidden layer
    return common_layer('softmax_linear', local4, [192, NUM_CLASSES], tensorflow.matmul, tensorflow.add,
                        lambda x, name: x)


# calculate the loss
def calculate_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = math_ops.cast(labels, dtypes.int64)
    cross_entropy = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tensorflow.reduce_mean(cross_entropy, name='cross_entropy')

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    # see #create_variable_with_weight_decay
    tensorflow.add_to_collection('losses', cross_entropy_mean)
    return tensorflow.add_n(tensorflow.get_collection('losses'), name='total_loss')


# calculate the evaluation accuracy during training
def evaluation(logits, labels):
    logits = math_ops.cast(logits, dtypes.float32)
    labels = math_ops.cast(labels, dtypes.int64)
    correct = tensorflow.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tensorflow.reduce_sum(tensorflow.cast(correct, tensorflow.int32))


# setup gradient decent optimizer
def training(loss, global_step, initial_learning_rate=0.1, learning_rate_decay_factor=0.1, num_epoch_per_decay=500):
    num_step_per_decay = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE * num_epoch_per_decay
    learning_rate = learning_rate_decay.exponential_decay(
        initial_learning_rate, global_step, num_step_per_decay, learning_rate_decay_factor)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    return optimizer.minimize(loss, global_step=global_step)


# setup summary and checkpoints
