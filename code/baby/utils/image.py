import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy
from tensorflow.python.keras._impl.keras.preprocessing.image import ImageDataGenerator


def read(path=""):
    image = Image.open(path)

    # padding
    long_side = max(image.size)
    horizontal_left_padding = int((long_side - image.size[0]) / 2)
    horizontal_right_padding = long_side - image.size[0] - horizontal_left_padding
    vertical_top_padding = int((long_side - image.size[1]) / 2)
    vertical_bottom_padding = long_side - image.size[1] - vertical_top_padding
    image = ImageOps.expand(
        image, (
            horizontal_left_padding,
            vertical_top_padding,
            horizontal_right_padding,
            vertical_bottom_padding
        ), fill=255)

    # thumbnail
    image.thumbnail([28, 28])
    image.convert("LA")
    image = 1 - numpy.asarray(image, dtype=numpy.float32) / 255
    image = numpy.dot(image[..., :3], [0.299, 0.587, 0.114])
    image = numpy.asarray(image, dtype=numpy.float32)
    return numpy.reshape(image, (1, 28, 28, 1))


def augment(features, labels, multiply=3):
    more_features = numpy.asarray(numpy.reshape([], (-1, 28, 28, 1)), features.dtype)
    more_labels = numpy.asarray(numpy.reshape([], (-1, 10)), labels.dtype)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2)
    datagen.fit(features)
    for _feature_batch, _label_batch in datagen.flow(features, labels, batch_size=100):
        feature_length = more_features.shape[0]
        if feature_length % 1000 == 0:
            print('{} generated'.format(feature_length))
        if feature_length >= features.shape[0] * multiply:
            break
        more_features = numpy.concatenate((more_features, _feature_batch))
        more_labels = numpy.concatenate((more_labels, _label_batch))

    return more_features, more_labels
