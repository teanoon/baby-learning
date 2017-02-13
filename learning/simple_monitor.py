from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.summary import summary
import tensorflow

import math
import time

LOG_DIR = '/tmp/simple_mnist_logs'

NUM_CLASSES = 10
BATCH_SIZE = 100
# TODO dynamically read from mnist data set
MNIST_IMAGE_SIZE = 28 * 28
MAX_STEPS = 2000


# TODO declare type info
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tensorflow.name_scope('summaries'):
        mean = tensorflow.reduce_mean(var)
        summary.scalar('mean', mean)
        # TODO why within named scope?
        with tensorflow.name_scope('stddev'):
            stddev = tensorflow.sqrt(tensorflow.reduce_mean(tensorflow.square(var - mean)))
        summary.scalar('stddev', stddev)
        summary.scalar('max', tensorflow.reduce_max(var))
        summary.scalar('min', tensorflow.reduce_min(var))
        summary.histogram('histogram', var)


def layer(name, current_layer, current_layer_units, next_layer_units, activate=tensorflow.nn.relu):
    with tensorflow.name_scope(name):
        with tensorflow.name_scope('weights'):
            weights = tensorflow.Variable(
                tensorflow.truncated_normal(
                    [current_layer_units, next_layer_units], stddev=1.0 / math.sqrt(current_layer_units)),
                name="weights")
            variable_summaries(weights)
        with tensorflow.name_scope('biases'):
            biases = tensorflow.Variable(
                tensorflow.zeros([next_layer_units]),
                name='biases')
            variable_summaries(biases)

        activations = activate(tensorflow.matmul(current_layer, weights) + biases)
        summary.histogram('activations', activations)

    return activations


def inference(images, hidden1_units=128, hidden2_units=32):
    """Build the MNIST model up to where it may be used for inference.
    Args:
      images: Images placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    hidden1 = layer('hidden1', images, MNIST_IMAGE_SIZE, hidden1_units)
    hidden2 = layer('hidden2', hidden1, hidden1_units, hidden2_units)
    return layer('softmax_linear', hidden2, hidden2_units, NUM_CLASSES, activate=tensorflow.identity)


def calculate_loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tensorflow.to_int64(labels)
    cross_entropy = tensorflow.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="x_entropy")
    return tensorflow.reduce_mean(cross_entropy, name="x_entropy_mean")


def training(loss, learning_rate):
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tensorflow.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tensorflow.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    return optimizer.minimize(loss, global_step=global_step)


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size, NUM_Classes].
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size) that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tensorflow.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    accuracy = tensorflow.reduce_sum(tensorflow.cast(correct, tensorflow.int32))
    summary.scalar('accuracy', accuracy)
    return accuracy


def do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      session: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      images_placeholder: The images placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of images and labels to evaluate, from input_data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += session.process(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))


def fill_feed_dict(data_set, images_placeholder, labels_placeholder):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
        data_set: The set of images and labels, from input_data.read_data_sets()
        images_placeholder: The images placeholder, from placeholder_inputs().
        labels_placeholder: The labels placeholder, from placeholder_inputs().
    Returns:
        feed_dict: The feed dictionary mapping from placeholders to values.
    """
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE, False)
    return {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed
    }


def run():
    data_set = input_data.read_data_sets("resources")

    images_placeholder = tensorflow.placeholder(tensorflow.float32, shape=(BATCH_SIZE, MNIST_IMAGE_SIZE))
    labels_placeholder = tensorflow.placeholder(tensorflow.int32, shape=BATCH_SIZE)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder, hidden1_units=128, hidden2_units=32)

    with tensorflow.name_scope('GradientDescent'):
        # Add to the Graph the Ops for loss calculation.
        loss = calculate_loss(logits, labels_placeholder)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, learning_rate=0.01)

    # setup complete
    # create a session to execute those calculations
    session = tensorflow.Session()

    # Merge all the summaries and write them out to /tmp/train
    merged = summary.merge_all()
    train_writer = summary.FileWriter('%s/train' % LOG_DIR, session.graph)
    test_writer = summary.FileWriter('%s/test' % LOG_DIR)

    # initial variables
    init = tensorflow.global_variables_initializer()
    session.run(init)

    # start training loop
    start_at = time.time()
    for step in range(MAX_STEPS):
        # TODO why
        # Fill a feed dictionary with the actual set of images and labels
        # for this particular training step.
        feed_dict = fill_feed_dict(data_set.train, images_placeholder, labels_placeholder)
        test_feed_dict = fill_feed_dict(data_set.test, images_placeholder, labels_placeholder)

        # originally we only need train_op per loop
        # we also want to observe loss for logging
        if step % 100 == 0:
            run_options = tensorflow.RunOptions(trace_level=tensorflow.RunOptions.FULL_TRACE)
            run_metadata = tensorflow.RunMetadata()
            log, _ = session.run(
                [merged, train_op],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%d' % step)
        else:
            log, _ = session.run([merged, train_op], feed_dict=feed_dict)
        train_writer.add_summary(log, step)

        if step % 100 == 0 or step + 1 == MAX_STEPS:
            log, loss_value = session.run([merged, loss], feed_dict=test_feed_dict)
            test_writer.add_summary(log, step)
            passed = time.time() - start_at
            start_at = time.time()
            print('Step %d: loss = %f (%f sec)' % (step, loss_value, passed))

        if step + 1 == MAX_STEPS:
            do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set.train)
            do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set.test)
            do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set.validation)

    train_writer.close()
    test_writer.close()

run()
