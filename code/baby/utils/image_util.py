import PIL.Image as Image
import PIL.ImageOps as ImageOps
import numpy


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
    return numpy.reshape(image, (1, 784))
