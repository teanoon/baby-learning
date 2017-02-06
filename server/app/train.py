import os
import time

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.framework import dtypes

from server.app import service


RESOURCES = os.path.join(os.path.dirname(__file__), '../resources')
CHECKPOINT_FOLDER = os.path.join(os.path.dirname(__file__), '../resources/simple_mnist_checkpoints/')
CHECKPOINT = os.path.join(os.path.dirname(__file__), '../resources/simple_mnist_checkpoints/model.ckpt')
LOG_DIR = '/tmp/simple_mnist_logs'

# Input
data_set = input_data.read_data_sets(RESOURCES)

# train variables and ops
label_placeholder = tensorflow.placeholder(dtypes.int64, shape=None)
image_placeholder = tensorflow.placeholder(dtypes.float32, shape=[None, 28, 28, 1])
logits = service.inference(image_placeholder)
total_loss = service.calculate_loss(logits, label_placeholder)
global_step = service.tensorflow.Variable(0, name='global_step', trainable=False)
train = service.training(
    total_loss, global_step, initial_learning_rate=0.1, learning_rate_decay_factor=0.1, num_epochs_per_decay=1)

# test/validate
accuracy = service.validate(logits, label_placeholder)

# initialize
session = tensorflow.Session()
init = tensorflow.global_variables_initializer()
session.run(init)

# Add ops to save and restore all the variables.
if os.path.exists(CHECKPOINT_FOLDER):
    for root, dirs, files in os.walk(CHECKPOINT_FOLDER):
        for name in files:
            os.remove(os.path.join(root, name))
saver = tensorflow.train.Saver()

# Merge all the summaries and write them out to /tmp/train
merged = tensorflow.summary.merge_all()
train_writer = tensorflow.summary.FileWriter('%s/train' % LOG_DIR, session.graph)
test_writer = tensorflow.summary.FileWriter('%s/test' % LOG_DIR, session.graph)
validation_writer = tensorflow.summary.FileWriter('%s/validation' % LOG_DIR, session.graph)

start_time_stamp = time.time()
print("starting")
for epoch in range(service.NUM_EPOCHS):
    print("Start epoch#%s ..." % (epoch + 1))
    for step in range(service.NUM_STEPS_PER_EPOCH_FOR_TRAIN):
        NUMBER = epoch * service.NUM_STEPS_PER_EPOCH_FOR_TRAIN + step + 1

        # Train
        images, labels = data_set.train.next_batch(service.BATCH_SIZE)
        images = session.run(tensorflow.reshape(images, shape=[service.BATCH_SIZE, 28, 28, 1]))
        summary, _train = session.run([merged, train], feed_dict={
            label_placeholder: labels,
            image_placeholder: images
        })
        train_writer.add_summary(summary, NUMBER)

        # Test
        if step + 1 == service.NUM_STEPS_PER_EPOCH_FOR_TRAIN:
            images = data_set.test.images
            images = session.run(tensorflow.reshape(images, shape=[images.shape[0], 28, 28, 1]))
            summary, _test = session.run([merged, accuracy], feed_dict={
                label_placeholder: data_set.test.labels,
                image_placeholder: images
            })
            test_writer.add_summary(summary, NUMBER)

        # Validate
        if NUMBER % 10 == 0 or step + 1 == service.NUM_STEPS_PER_EPOCH_FOR_TRAIN:
            images = data_set.validation.images
            images = session.run(tensorflow.reshape(images, shape=[images.shape[0], 28, 28, 1]))
            summary, _validate = session.run([merged, accuracy], feed_dict={
                label_placeholder: data_set.validation.labels,
                image_placeholder: images
            })
            validation_writer.add_summary(summary, NUMBER)

saver.save(session, CHECKPOINT)
train_writer.close()
test_writer.close()
validation_writer.close()
