import itertools
import os
import uuid

from hyperopt import hp, fmin, tpe
from hyperopt.mongoexp import MongoTrials
from tensorflow.examples.tutorials.mnist import input_data

from server.app.service import SessionRunner


def training(args):
    data_set = input_data.read_data_sets(os.path.join(os.path.dirname(__file__), '../resources'))

    losses = []
    validation_accuracies = []
    with SessionRunner(args) as runner:
        loss, train, accuracy = runner.draw_graph()
        # run
        for epoch, step in itertools.product(range(runner.epochs), range(runner.steps)):
            if step == 0:
                print('start epoch %d' % (epoch + 1))

            # train
            images, labels = data_set.train.next_batch(runner.batch_size)
            _loss, _train = runner.process([loss, train], images, labels)
            losses.append(_loss)

            # validate
            if not step + 1 == runner.steps:
                continue
            validation_accuracy = runner.process(accuracy, data_set.validation.images, data_set.validation.labels)
            validation_accuracies.append(validation_accuracy)

        # objective value
        test_accuracy = runner.process(accuracy, data_set.test.images, data_set.test.labels)
        return 1 - test_accuracy


space = {
    'batch_size': hp.quniform('batch_size', 1, 300, 1),
    'epochs': hp.quniform('epochs', 1, 2, 1),
    'initial_learning_rate': hp.uniform('initial_learning_rate', 0.01, 0.5),
    'learning_rate_decay_factor': hp.uniform('learning_rate_decay_factor', 0.01, 0.1),
    'num_epochs_per_decay': hp.quniform('num_epochs_per_decay', 1, 20, 1)
}

# TODO use MongoTrials
trials = MongoTrials('mongo://mongo:27017/mnist/jobs', exp_key='case-1')
best = fmin(
    training,
    space=dict(space.items(), **{'example_size': 500}),
    algo=tpe.suggest,
    trials=trials,
    max_evals=10)
print(best)

# final_errors = training(dict(best.items(), **{'example_size': 55000}))
# print('test accuracy: %f' % (1 - final_errors))
