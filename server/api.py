import PIL.ImageOps
import numpy
import tensorflow
from flask import Flask, jsonify, request
from PIL import Image

from server import service

# restore the train data
image_placeholder = tensorflow.placeholder(
    tensorflow.float32,
    shape=[1, 28, 28, 1])
session = tensorflow.Session()

# restore trained data
logits = service.inference(image_placeholder)
labels = service.softmax(logits)
saver = tensorflow.train.Saver()
saver.restore(session, "resources/simple_mnist_checkpoints/model.ckpt")


def read_image(_input):
    # TODO we won't need greyscale in the future
    image = Image.open(_input).convert('L')

    # padding
    long_side = max(image.size)
    horizontal_left_padding = int((long_side - image.size[0]) / 2)
    horizontal_right_padding = long_side - image.size[0] - horizontal_left_padding
    vertical_top_padding = int((long_side - image.size[1]) / 2)
    vertical_bottom_padding = long_side - image.size[1] - vertical_top_padding
    image = PIL.ImageOps.expand(
        image,
        (horizontal_left_padding, vertical_top_padding, horizontal_right_padding, vertical_bottom_padding),
        fill=255)

    image.thumbnail((28, 28))

    # TODO maybe we need a transpose here?
    image = numpy.asarray(image, dtype=numpy.uint8).reshape((1, 28, 28, 1))
    return (255 - image) / 255.0


def convolutional(_input):
    return session.run(labels, feed_dict={image_placeholder: _input}).flatten().tolist()

# start the app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route("/")
def hello():
    return jsonify(results={'say': 'hello world!'})


@app.route("/recognize", methods=['POST'])
def recognize():
    _input = read_image(request.files['file'])
    output = convolutional(_input)
    return jsonify(output=output, hit=str(numpy.argmax(output)))

if __name__ == "__main__":
    app.run(host='0.0.0.0')
