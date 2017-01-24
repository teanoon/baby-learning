import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

from app import service

validation_set = input_data.read_data_sets('../resources').validation
image_placeholder = tensorflow.placeholder(tensorflow.float32, shape=[service.BATCH_SIZE, 28, 28, 1])
label_placeholder = tensorflow.placeholder(tensorflow.int64, shape=service.BATCH_SIZE)

# evaluation step
logits = service.inference(image_placeholder)
accuracy = service.evaluation(logits, label_placeholder)

# initialize
session = tensorflow.Session()
init = tensorflow.global_variables_initializer()
session.run(init)

# Add ops to save and restore all the variables.
saver = tensorflow.train.Saver()
saver.restore(session, '../resources/simple_mnist_checkpoints/model.ckpt')

# run one epoch of eval.
true_count = 0
steps_per_epoch = validation_set.num_examples // service.BATCH_SIZE
num_examples = steps_per_epoch * service.BATCH_SIZE
print('start evaluate')
for _step in range(steps_per_epoch):
    validation_images, validation_labels = validation_set.next_batch(service.BATCH_SIZE)
    validation_images = session.run(tensorflow.reshape(validation_images, [service.BATCH_SIZE, 28, 28, 1]))
    true_count += session.run(accuracy, feed_dict={
        label_placeholder: validation_labels,
        image_placeholder: validation_images
    })
precision = float(true_count) / num_examples * 100
print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))
