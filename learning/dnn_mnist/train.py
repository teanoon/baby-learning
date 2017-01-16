import time
import tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables

from dnn_mnist import service

# Input
data_set = service.input_with_type('train', '../resources')
label_placeholder = array_ops.placeholder(dtypes.int64, shape=service.BATCH_SIZE)
image_placeholder = array_ops.placeholder(dtypes.float16, shape=[service.BATCH_SIZE, 28, 28, 1])

# Construct learning graph
logits = service.inference(image_placeholder)
loss = service.calculate_loss(logits, label_placeholder)
accuracy = service.evaluation(logits, label_placeholder)
global_step = service.tensorflow.Variable(0, name='global_step', trainable=False)
train = service.training(loss, global_step)

# initialize
session = tensorflow.Session()
init = variables.global_variables_initializer()
session.run(init)

# train
start_time = time.time()
print("starting")
for step in range(10):
    # TODO use stream the inputs to improve efficiency
    images, labels = data_set.next_batch(service.BATCH_SIZE)
    next_batch_time = time.time()
    images = session.run(array_ops.reshape(images, [service.BATCH_SIZE, 28, 28, 1]))
    feed_dict = {
        label_placeholder: labels,
        image_placeholder: images
    }
    feed_time = time.time()
    session.run(train, feed_dict=feed_dict)

    if step % 100 == 99 or step + 1 == 10:
        loss_value, accuracy_value = session.run([loss, accuracy], feed_dict=feed_dict)
        print(loss_value)
        print(accuracy_value)

    print("step %s: next_batch - %f feed - %f, total - %f"
          % (step + 1, next_batch_time - start_time, feed_time - next_batch_time, time.time() - start_time))
    start_time = time.time()
