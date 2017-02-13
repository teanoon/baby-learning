import uuid

import tensorflow
from tensorflow.python.framework import dtypes

from server.app import model

LOG_DIR = '/tmp/simple_mnist_logs'


class SessionRunner:
    def __init__(self, args):
        example_size = args.get('example_size')
        self.batch_size = int(args.get('batch_size'))
        self.steps = int(example_size / self.batch_size)
        self.epochs = int(args.get('epochs'))
        self.initial_learning_rate = args.get('initial_learning_rate')
        self.learning_rate_decay_factor = args.get('learning_rate_decay_factor')
        num_epochs_per_decay = args.get('num_epochs_per_decay')
        self.num_steps_per_decay = self.steps * num_epochs_per_decay

        self.session = tensorflow.Session()
        self.label_placeholder = tensorflow.placeholder(dtypes.int64, shape=None)
        self.image_placeholder = tensorflow.placeholder(dtypes.float32, shape=[None, 28, 28, 1])

        self.train_writer = tensorflow.summary.FileWriter('%s/train' % LOG_DIR, self.session.graph)
        self.test_writer = tensorflow.summary.FileWriter('%s/test' % LOG_DIR, self.session.graph)
        self.validation_writer = tensorflow.summary.FileWriter('%s/validation' % LOG_DIR, self.session.graph)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def draw_graph(self):
        with tensorflow.variable_scope("case-{}".format(uuid.uuid4())):
            global_step = tensorflow.Variable(0, name='global_step', trainable=False)
            logits = model.inference(self.image_placeholder)
            loss = model.loss(logits, self.label_placeholder)
            train = model.training(
                loss, global_step,
                initial_learning_rate=self.initial_learning_rate,
                learning_rate_decay_factor=self.learning_rate_decay_factor,
                num_steps_per_decay=self.num_steps_per_decay)
            accuracy = model.validate(logits, self.label_placeholder)

        init = tensorflow.global_variables_initializer()
        self.session.run(init)

        return loss, train, accuracy

    def process(self, opts, _images, _labels):
        _images = tensorflow.reshape(_images, shape=[_images.shape[0], 28, 28, 1])
        _images = self.session.run(_images)
        return self.session.run(opts, feed_dict={
            self.label_placeholder: _labels,
            self.image_placeholder: _images
        })

    def close(self):
        self.session.close()
        tensorflow.reset_default_graph()
        self.train_writer.close()
        self.test_writer.close()
        self.validation_writer.close()
