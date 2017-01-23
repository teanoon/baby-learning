import time

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

from server import service

LOG_DIR = '/tmp/simple_mnist_logs'

# Input
data_set = input_data.read_data_sets('../resources').train
test_set = input_data.read_data_sets('../resources').test
validation_set = input_data.read_data_sets('../resources').validation
label_placeholder = tensorflow.placeholder(tensorflow.int64, shape=service.BATCH_SIZE)
image_placeholder = tensorflow.placeholder(tensorflow.float32, shape=[service.BATCH_SIZE, 28, 28, 1])

# Construct learning graph
logits = service.inference(image_placeholder)
total_loss = service.calculate_loss(logits, label_placeholder)
global_step = service.tensorflow.Variable(0, name='global_step', trainable=False)
train = service.training(total_loss, global_step)
# evaluation step
accuracy = service.evaluation(logits, label_placeholder)

# initialize
session = tensorflow.Session()
init = tensorflow.global_variables_initializer()
session.run(init)

# Add ops to save and restore all the variables.
saver = tensorflow.train.Saver()

# Merge all the summaries and write them out to /tmp/train
merged = tensorflow.summary.merge_all()
train_writer = tensorflow.summary.FileWriter('%s/train' % LOG_DIR, session.graph)

# train
start_time = time.time()
print("starting")
for step in range(service.MAX_STEPS):
    # Get next batch
    images, labels = data_set.next_batch(service.BATCH_SIZE)
    images = session.run(tensorflow.reshape(images, [service.BATCH_SIZE, 28, 28, 1]))
    feed_dict = {
        label_placeholder: labels,
        image_placeholder: images
    }
    feed_time = time.time()

    # train
    run_options = tensorflow.RunOptions(trace_level=tensorflow.RunOptions.FULL_TRACE)
    run_metadata = tensorflow.RunMetadata()
    log, _ = session.run([merged, train], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
    train_writer.add_run_metadata(run_metadata, 'step%d' % (step + 1))
    train_writer.add_summary(log, step + 1)

    print("step %s: feed - %f, total - %f" % (step + 1, feed_time - start_time, time.time() - start_time))

    if step % 100 == 99:
        loss_value = session.run(total_loss, feed_dict=feed_dict)
        print("loss: %f" % loss_value)

    start_time = time.time()

saver.save(session, 'resources/simple_mnist_checkpoints/model.ckpt')
train_writer.close()
