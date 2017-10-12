import os
import uuid

import pymongo
import tensorflow
from hyperopt import Trials
from tensorflow.python.framework import dtypes

from server.app import model

CHECKPOINT = os.path.join(os.path.dirname(__file__), '../checkpoints/simple_mnist_checkpoints/model.ckpt')
LOG_DIR = os.path.join(os.path.dirname(__file__), '../log/simple_mnist_logs')


class SessionRunner:
    def __init__(self, args):
        self.is_optimizing = args.get('is_optimizing')
        self.exp_key = args.get('exp_key')
        example_size = args.get('example_size')
        self.batch_size = int(args.get('batch_size'))
        self.steps = int(example_size / self.batch_size)
        self.epochs = int(args.get('epochs'))
        self.initial_learning_rate = args.get('initial_learning_rate')
        self.learning_rate_decay_factor = args.get('learning_rate_decay_factor')
        num_epochs_per_decay = args.get('num_epochs_per_decay')
        self.num_steps_per_decay = self.steps * num_epochs_per_decay

        self.session = tensorflow.Session()
        with tensorflow.variable_scope("case-{}-{}".format(self.exp_key, uuid.uuid4())):
            self._global_step = tensorflow.Variable(0, name='global_step', trainable=False)
            self.label_placeholder = tensorflow.placeholder(dtypes.int64, shape=None)
            self.image_placeholder = tensorflow.placeholder(dtypes.float32, shape=[None, 28, 28, 1])
            logits = model.inference(self.image_placeholder)
            loss = model.loss(logits, self.label_placeholder)
            self.train = model.training(
                loss, self._global_step,
                initial_learning_rate=self.initial_learning_rate,
                learning_rate_decay_factor=self.learning_rate_decay_factor,
                num_steps_per_decay=self.num_steps_per_decay)
            self.accuracy = model.validate(logits, self.label_placeholder)

            init = tensorflow.global_variables_initializer()
            self.session.run(init)

            self.merged = tensorflow.summary.merge_all()
            self.train_writer = tensorflow.summary.FileWriter('%s/train' % LOG_DIR, self.session.graph)
            self.validation_writer = tensorflow.summary.FileWriter('%s/validation' % LOG_DIR, self.session.graph)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def global_step(self):
        return self.session.run(self._global_step)

    def checkpoint(self):
        if self.is_optimizing:
            pass
        folder = os.path.join(
            os.path.dirname(__file__), '../checkpoints/{}/'.format(self.exp_key))
        if not os.path.exists(folder):
            os.mkdir(folder)
        path = os.path.join(folder, 'model.ckpt')
        saver = tensorflow.train.Saver()
        saver.save(self.session, path)

    def process(self, opts, _images, _labels):
        _images = tensorflow.reshape(_images, shape=[_images.shape[0], 28, 28, 1])
        _images = self.session.run(_images)
        return self.session.run(opts, feed_dict={
            self.label_placeholder: _labels,
            self.image_placeholder: _images
        })

    def add_train_summary(self, summary, step):
        self.train_writer.add_summary(summary, step)

    def add_validation_summary(self, summary, step):
        self.validation_writer.add_summary(summary, step)

    def close(self):
        self.checkpoint()
        self.session.close()
        tensorflow.reset_default_graph()
        self.train_writer.close()
        self.validation_writer.close()


class OptimizationTracker:
    def __init__(self, database_name):
        self.client = pymongo.MongoClient("mongo", 27017)
        self.collection = self.client[database_name].trials

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def store(self, trials):
        for trial in trials:
            self.collection.replace_one({'exp_key': trial['exp_key'], 'tid': trial['tid']}, trial, upsert=True)

    def restore(self, exp_key):
        trials = Trials(exp_key=exp_key)
        for item in self.collection.find({'exp_key': exp_key}):
            trials.insert_trial_doc(item)
        trials.refresh()
        return trials

    def remove(self, exp_key):
        self.collection.delete_many({'exp_key': exp_key})

    def remove_collection(self):
        self.collection.remove()

    def close(self):
        self.client.close()
