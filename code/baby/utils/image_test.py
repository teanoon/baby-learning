from unittest import TestCase

import numpy

from utils import image


class ImageTest(TestCase):
    def test_augment(self):
        features = numpy.asarray(numpy.random.uniform(0, 10, (10, 28, 28, 1)), numpy.float32)
        labels = numpy.asarray(numpy.random.uniform(0, 10, (10, 10)), numpy.int32)
        more_features, more_labels = image.augment(features[:10], labels[:10], multiply=20)
        self.assertEqual(20*10, more_features.shape[0])
        self.assertEqual(20*10, more_labels.shape[0])
        self.assertEqual(numpy.float32, more_features.dtype)
        self.assertEqual(numpy.int32, more_labels.dtype)
