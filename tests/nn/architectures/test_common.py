import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import common


class CommonTests(tf.test.TestCase):
    def test_intermediate_feature(self):
        intermediate_feature = common.IntermediateFeature(
            tensor=tf.zeros((1, 1, 1, 1)), stride=4
        )
        self.assertEqual(intermediate_feature.scale, 0.25)
