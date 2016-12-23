# Import data
from tensorflow.examples.tutorials.mnist import input_data

import math
import time

import tensorflow as tf


NUM_CLASSES = 10
BATCH_SIZE = 100
# TODO dynamically read from mnist data set
MNIST_IMAGE_SIZE = 28 * 28
MAX_STEPS = 2000


def inference(images, hidden1_units=128, hidden2_units=32):
    """Build the MNIST model up to where it may be used for inference.
    Args:
      images: Images placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.
    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([MNIST_IMAGE_SIZE, hidden1_units], stddev=1.0 / math.sqrt(MNIST_IMAGE_SIZE)),
            name="weights")
        biases = tf.Variable(
            tf.zeros([hidden1_units]),
            name='biases')

        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units], stddev=1.0 / math.sqrt(hidden1_units)),
            name="weights")
        biases = tf.Variable(
            tf.zeros([hidden2_units]),
            name="biases")

        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([hidden2_units, NUM_CLASSES], stddev=1.0 / math.sqrt(hidden2_units)),
            name="weights")
        biases = tf.Variable(
            tf.zeros([NUM_CLASSES]),
            name="biases")

        logits = tf.matmul(hidden2, weights) + biases

    return logits


def calculate_loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].
    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name="x_entropy")
    return tf.reduce_mean(cross_entropy, name="x_entropy_mean")


def training(loss, learning_rate):
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
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
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))


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
        true_count += session.run(eval_correct, feed_dict=feed_dict)
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

    with tf.Graph().as_default():
        images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, MNIST_IMAGE_SIZE))
        labels_placeholder = tf.placeholder(tf.int32, shape=BATCH_SIZE)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(images_placeholder, hidden1_units=128, hidden2_units=32)

        # Add to the Graph the Ops for loss calculation.
        loss = calculate_loss(logits, labels_placeholder)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, learning_rate=0.01)

        # setup complete
        # create a session to execute those calculations
        session = tf.Session()

        # initial variables
        init = tf.initialize_all_variables()
        session.run(init)

        # start training loop
        start_at = time.time()
        for step in range(MAX_STEPS):
            # TODO why
            # Fill a feed dictionary with the actual set of images and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_set.train, images_placeholder, labels_placeholder)

            # originally we only need train_op per loop
            # we also want to observe loss for logging
            _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

            if step % 100 == 0:
                passed = time.time() - start_at
                start_at = time.time()
                print('Step %d: loss = %f (%f sec)' % (step, loss_value, passed))

            if step + 1 == MAX_STEPS:
                do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set.train)
                do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set.test)
                do_evaluate(session, eval_correct, images_placeholder, labels_placeholder, data_set.validation)


run()
