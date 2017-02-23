import os

import numpy
import tensorflow

from server.learning.word_embedding import model

TEXT_FILE = os.path.join(os.path.dirname(__file__), '../../resources/text8')
VOCABULARY_SIZE = 50000
EMBEDDING_SIZE = 1000
BATCH_SIZE = 100
NUM_STEPS = 10000

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
# Random set of words to evaluate similarity on.
VALID_SIZE = 16
# Only pick dev samples in the head of the distribution.
VALID_WINDOW = 100
# Number of negative examples to sample.
NUM_SAMPLED = 64


def train():
    # data
    data, dictionary_by_words, dictionary_by_popular = model.read_data(TEXT_FILE, VOCABULARY_SIZE)

    # variables
    train_inputs = tensorflow.placeholder(tensorflow.int32, shape=[BATCH_SIZE])
    train_labels = tensorflow.placeholder(tensorflow.int32, shape=[BATCH_SIZE, 1])
    embeddings = tensorflow.Variable(
        tensorflow.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))

    valid_examples = numpy.random.choice(VALID_WINDOW, VALID_SIZE, replace=False)
    valid_data_set = tensorflow.constant(valid_examples, dtype=tensorflow.int32)

    # ops
    loss = model.loss(train_inputs, train_labels, embeddings, VOCABULARY_SIZE, EMBEDDING_SIZE, NUM_SAMPLED)
    optimize = model.optimize(loss)
    validate = model.validate(embeddings, valid_data_set)

    init = tensorflow.global_variables_initializer()
    session = tensorflow.Session()
    session.run(init)

    data_index = 0
    total_loss = 0
    for step in range(NUM_STEPS):
        batch_input, batch_labels, data_index = model.generate_batch(
            data, data_index, batch_size=BATCH_SIZE, num_skips=2, skip_window=1)
        _, _loss = session.run([optimize, loss], feed_dict={
            train_inputs: batch_input,
            train_labels: batch_labels})
        total_loss += _loss

        if (step + 1) % 2000 == 0:
            total_loss /= 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step %s: %f" % (step + 1, total_loss))
            total_loss = 0

        if (step + 1) % 10000 == 0:
            _validate = session.run(validate)
            for i in range(VALID_SIZE):
                valid_word = dictionary_by_popular[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-_validate[i, :]).argsort()[1:top_k + 1]
                log_str = "Nearest to \"%s\":" % valid_word
                for k in range(top_k):
                    close_word = dictionary_by_popular[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)

train()
