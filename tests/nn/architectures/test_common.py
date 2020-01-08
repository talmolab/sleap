import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.architectures import common


class CommonTests(tf.test.TestCase):
    def test_intermediate_feature(self):
        intermediate_feature = common.IntermediateFeature(
            tensor=tf.zeros((1, 1, 1, 1)), stride=4
        )
        self.assertEqual(intermediate_feature.scale, 0.25)
