import io
import json
import os
import unittest

import datetime

from hyperopt import Trials

from server.app.service import OptimizationTracker

NUMBER_EIGHT_PIC = os.path.join(os.path.dirname(__file__), '../resources/8.jpg')


class OptimizationTrackerTest(unittest.TestCase):
    def setUp(self):
        self.data = [{'tid': 0, 'result': {'loss': 0.902, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 0, 'vals': {'batch_size': [108.0], 'num_epochs_per_decay': [4.0], 'epochs': [1.0], 'learning_rate_decay_factor': [0.03364510178322278], 'initial_learning_rate': [0.29030387216338477]}, 'idxs': {'batch_size': [0], 'num_epochs_per_decay': [0], 'epochs': [0], 'learning_rate_decay_factor': [0], 'initial_learning_rate': [0]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 55, 18, 344000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 55, 29, 623000), 'spec': None}, {'tid': 1, 'result': {'loss': 0.8866, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 1, 'vals': {'batch_size': [95.0], 'num_epochs_per_decay': [3.0], 'epochs': [1.0], 'learning_rate_decay_factor': [0.1650297442363764], 'initial_learning_rate': [0.32997423156951206]}, 'idxs': {'batch_size': [1], 'num_epochs_per_decay': [1], 'epochs': [1], 'learning_rate_decay_factor': [1], 'initial_learning_rate': [1]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 55, 29, 628000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 55, 40, 863000), 'spec': None}, {'tid': 2, 'result': {'loss': 0.876, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 2, 'vals': {'batch_size': [81.0], 'num_epochs_per_decay': [5.0], 'epochs': [1.0], 'learning_rate_decay_factor': [0.08802834278612955], 'initial_learning_rate': [0.4910721187947583]}, 'idxs': {'batch_size': [2], 'num_epochs_per_decay': [2], 'epochs': [2], 'learning_rate_decay_factor': [2], 'initial_learning_rate': [2]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 55, 40, 868000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 55, 52, 221000), 'spec': None}, {'tid': 3, 'result': {'loss': 0.9042, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 3, 'vals': {'batch_size': [175.0], 'num_epochs_per_decay': [19.0], 'epochs': [1.0], 'learning_rate_decay_factor': [0.1651893658870274], 'initial_learning_rate': [0.318026249380799]}, 'idxs': {'batch_size': [3], 'num_epochs_per_decay': [3], 'epochs': [3], 'learning_rate_decay_factor': [3], 'initial_learning_rate': [3]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 55, 52, 225000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 56, 2, 856000), 'spec': None}, {'tid': 4, 'result': {'loss': 0.8793, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 4, 'vals': {'batch_size': [233.0], 'num_epochs_per_decay': [19.0], 'epochs': [1.0], 'learning_rate_decay_factor': [0.03716971154734972], 'initial_learning_rate': [0.20177973164688473]}, 'idxs': {'batch_size': [4], 'num_epochs_per_decay': [4], 'epochs': [4], 'learning_rate_decay_factor': [4], 'initial_learning_rate': [4]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 56, 2, 859000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 56, 13, 828000), 'spec': None}, {'tid': 5, 'result': {'loss': 0.7896, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 5, 'vals': {'batch_size': [80.0], 'num_epochs_per_decay': [11.0], 'epochs': [2.0], 'learning_rate_decay_factor': [0.1120101285956957], 'initial_learning_rate': [0.47955613746412756]}, 'idxs': {'batch_size': [5], 'num_epochs_per_decay': [5], 'epochs': [5], 'learning_rate_decay_factor': [5], 'initial_learning_rate': [5]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 56, 13, 832000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 56, 29, 512000), 'spec': None}, {'tid': 6, 'result': {'loss': 0.8865, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 6, 'vals': {'batch_size': [221.0], 'num_epochs_per_decay': [15.0], 'epochs': [1.0], 'learning_rate_decay_factor': [0.08595409264623936], 'initial_learning_rate': [0.4291889545667844]}, 'idxs': {'batch_size': [6], 'num_epochs_per_decay': [6], 'epochs': [6], 'learning_rate_decay_factor': [6], 'initial_learning_rate': [6]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 56, 29, 516000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 56, 42, 581000), 'spec': None}, {'tid': 7, 'result': {'loss': 0.9042, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 7, 'vals': {'batch_size': [285.0], 'num_epochs_per_decay': [10.0], 'epochs': [2.0], 'learning_rate_decay_factor': [0.1902629752826795], 'initial_learning_rate': [0.4029040292355903]}, 'idxs': {'batch_size': [7], 'num_epochs_per_decay': [7], 'epochs': [7], 'learning_rate_decay_factor': [7], 'initial_learning_rate': [7]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 56, 42, 585000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 56, 58, 216000), 'spec': None}, {'tid': 8, 'result': {'loss': 0.8573, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 8, 'vals': {'batch_size': [183.0], 'num_epochs_per_decay': [5.0], 'epochs': [2.0], 'learning_rate_decay_factor': [0.06056108575354093], 'initial_learning_rate': [0.3231672336993284]}, 'idxs': {'batch_size': [8], 'num_epochs_per_decay': [8], 'epochs': [8], 'learning_rate_decay_factor': [8], 'initial_learning_rate': [8]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 56, 58, 219000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 57, 15, 938000), 'spec': None}, {'tid': 9, 'result': {'loss': 0.902, 'status': 'ok'}, 'misc': {'workdir': None, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'tid': 9, 'vals': {'batch_size': [276.0], 'num_epochs_per_decay': [5.0], 'epochs': [2.0], 'learning_rate_decay_factor': [0.0368175259661329], 'initial_learning_rate': [0.4156446402578632]}, 'idxs': {'batch_size': [9], 'num_epochs_per_decay': [9], 'epochs': [9], 'learning_rate_decay_factor': [9], 'initial_learning_rate': [9]}}, 'exp_key': 'run1', 'book_time': datetime.datetime(2017, 2, 16, 7, 57, 15, 943000), 'owner': None, 'version': 0, 'state': 2, 'refresh_time': datetime.datetime(2017, 2, 16, 7, 57, 32, 681000), 'spec': None}]
        self.tracker = OptimizationTracker('mnist_test')

    def tearDown(self):
        self.tracker.remove_collection()
        self.tracker.close()

    def test_store_success(self):
        # given there is a Trials
        exp_key = 'run1'
        trials = Trials(exp_key=exp_key)
        [trials.insert_trial_doc(item) for item in self.data]
        trials.refresh()
        self.assertTrue(len(trials.trials) > 0)
        stored_trials = self.tracker.restore(exp_key)
        self.assertTrue(len(stored_trials.trials) == 0)

        # when save to the db
        self.tracker.store(trials.trials)

        # then restore the data
        trials = self.tracker.restore(exp_key)
        self.assertTrue(len(trials.trials) > 0)

    def test_store_success_to_avoid_duplicated(self):
        # given there is a stored Trials
        exp_key = 'run1'
        trials = Trials(exp_key=exp_key)
        [trials.insert_trial_doc(item) for item in self.data]
        trials.refresh()
        self.tracker.store(trials.trials)
        size = len(trials.trials)

        # when save to the db again
        self.tracker.store(trials.trials)

        # then restore the data
        trials = self.tracker.restore(exp_key)
        self.assertEqual(size, len(trials.trials))

    def test_remove_success(self):
        # given there is a stored Trials
        exp_key = 'run2'
        trials = Trials(exp_key=exp_key)
        for item in self.data:
            item['exp_key'] = exp_key
        [trials.insert_trial_doc(item) for item in self.data]
        trials.refresh()
        self.tracker.store(trials.trials)

        # when remove the trials from db
        self.tracker.remove(exp_key)

        # then restore is failed
        trials = self.tracker.restore(exp_key)
        self.assertTrue(len(trials.trials) == 0)

if __name__ == '__main__':
    unittest.main()
