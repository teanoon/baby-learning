import collections
import math

import numpy
import tensorflow


def read_data(text_file, vocabulary_size):
    with open(text_file) as file:
        words = tensorflow.compat.as_str(file.read()).split()

    # count common words from text
    # noinspection PyArgumentList
    count = dict(collections.Counter(words).most_common(vocabulary_size - 2))
    # count uncommon words
    count['UNK'] = len(words) - sum([num for _, num in count.items()])
    # ordered by popular and then alphabetic
    # _count = sorted(_count.items(), key=lambda item: (item[1], item[0]), reverse=True)
    dictionary_by_popular = [word for word, _ in count.items()]
    # keyed by words and valued by index in popular
    dictionary_by_words = dict((word, index + 1) for index, word in enumerate(dictionary_by_popular))
    data = [dictionary_by_words.get(word, 0) for word in words]
    del words
    return data, dictionary_by_words, dictionary_by_popular


# skip-gram model
def generate_batch(data, data_index, batch_size, num_skips, skip_window):
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    _batch_inputs = numpy.ndarray(shape=batch_size, dtype=numpy.int32)
    _batch_labels = numpy.ndarray(shape=(batch_size, 1), dtype=numpy.int32)
    _next_data_index = min(data_index + batch_size // num_skips, len(data) - 1)
    batch_index = 0
    for index, target in enumerate(data[data_index:_next_data_index + 1]):
        if index + 1 == skip_window:
            continue
        if index + skip_window == len(data):
            break
        contexts = data[index - skip_window:index] + data[index + 1:index + skip_window + 1]
        for context_index, context in enumerate(contexts):
            _batch_inputs[batch_index + context_index] = target
            _batch_labels[batch_index + context_index, 0] = context
        batch_index += len(contexts)

    return _batch_inputs, _batch_labels, _next_data_index


def loss(inputs, labels, embeddings, vocabulary_size, embedding_size, num_sampled):
    # inference
    weights = tensorflow.Variable(
        tensorflow.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    biases = tensorflow.Variable(
        tensorflow.zeros([vocabulary_size]))
    embed = tensorflow.nn.embedding_lookup(embeddings, inputs)

    # loss
    return tensorflow.reduce_mean(
        tensorflow.nn.nce_loss(
            weights, biases, labels, embed, num_sampled, vocabulary_size))


def optimize(_loss):
    return tensorflow.train.GradientDescentOptimizer(1.0).minimize(_loss)


def validate(embeddings, valid_data_set):
    normalize = tensorflow.sqrt(tensorflow.reduce_sum(tensorflow.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = tensorflow.divide(embeddings, normalize)
    valid_embeddings = tensorflow.nn.embedding_lookup(
        normalized_embeddings, valid_data_set)
    return tensorflow.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)
