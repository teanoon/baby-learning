import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from utils.serving_client import ServingClient


def main(argv):
    data_sets = input_data.read_data_sets(argv.input_dir, one_hot=True)
    test_data = data_sets.test
    request_data = [(test_data.images[0], test_data.labels[0])]
    client = ServingClient(argv.host, argv.port)
    response = client.request(argv.name, argv.signature_name, request_data)
    print(response)


if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('input-dir', '../data', 'Directory to read the training data.', short_name='input_dir')
    flags.DEFINE_string('host', 'serving', 'Tensorflow serving server host.')
    flags.DEFINE_string('port', '9000', 'Tensorflow serving server port.')
    flags.DEFINE_string('name', 'mnist', 'Tensorflow serving server current servable name.')
    flags.DEFINE_string('signature-name', 'serving_default', 'Tensorflow serving server current servable name.',
                        short_name='signature_name')
    flags.DEFINE_string('job-dir', '../job', 'Directory to put the model into.', short_name='job_dir')

    main(flags.FLAGS)
