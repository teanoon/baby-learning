import itertools
import json
import os

import yaml
from hyperopt import Trials, fmin, hp, tpe
from tensorflow.examples.tutorials.mnist import input_data

from server.app.service import SessionRunner, OptimizationTracker

PARAM_FILE_NAME = os.path.join(os.path.dirname(__file__), 'params.yml')
DATA_SET = input_data.read_data_sets(os.path.join(os.path.dirname(__file__), '../resources'))


def training(args):
    with SessionRunner(args) as runner:
        # run
        for epoch, step in itertools.product(range(runner.epochs), range(runner.steps)):
            if step == 0:
                print('start epoch %d' % (epoch + 1))

            global_step = runner.global_step()

            # train
            images, labels = DATA_SET.train.next_batch(runner.batch_size)
            _summary, _ = runner.process([runner.merged, runner.train], images, labels)
            runner.add_train_summary(_summary, global_step)

            # validate
            if not step + 1 == runner.steps:
                continue
            _summary, validation_accuracy = runner.process([runner.merged, runner.accuracy],
                                                           DATA_SET.validation.images, DATA_SET.validation.labels)
            runner.add_validation_summary(_summary, global_step)

        # objective value
        test_accuracy = runner.process(runner.accuracy, DATA_SET.test.images, DATA_SET.test.labels)
        return 1 - test_accuracy


with open(PARAM_FILE_NAME, 'r') as file:
    params = yaml.load(file)[-1]

space = {
    'batch_size': hp.quniform('batch_size', *params.get('batch_size_range')),
    'epochs': hp.quniform('epochs', *params.get('epoch_range')),
    'initial_learning_rate': hp.uniform('initial_learning_rate', *params.get('initial_learning_rate')),
    'learning_rate_decay_factor': hp.uniform('learning_rate_decay_factor', *params.get('learning_rate_decay_factor')),
    'num_epochs_per_decay': hp.quniform('num_epochs_per_decay', *params.get('num_epochs_per_decay'))
}

with OptimizationTracker('mnist') as tracker:
    trials = tracker.restore(params.get('exp_key'))
    best = fmin(
        training,
        space=dict(space.items(), **{'example_size': 500, 'exp_key': params.get('exp_key')}),
        algo=tpe.suggest,
        trials=trials,
        max_evals=params.get('max_evals'))
    tracker.store(trials.trials)
    print(best)

final_errors = training(dict(best.items(), **{'example_size': 500, 'exp_key': params.get('exp_key')}))
print('test accuracy: %f' % (1 - final_errors))
