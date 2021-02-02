import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import leap
from sleap.nn.config import LEAPConfig


class LeapTests(tf.test.TestCase):
    def test_leap_cnn_reference(self):
        # Reference implementation from the original paper.
        arch = leap.LeapCNN(
            filters=64,
            filters_rate=2,
            down_blocks=3,
            down_convs_per_block=3,
            up_blocks=3,
            up_interpolate=False,
            up_convs_per_block=2,
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 40)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 36)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 10768896)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 10768896)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 192, 192, 128))
        with self.subTest("number of intermediate features"):
            self.assertEqual(len(x_mid), 3)
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 8)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)
        with self.subTest("layer instance type by name"):
            self.assertIsInstance(
                model.get_layer("stack0_dec0_s8_to_s4_trans_conv"),
                tf.keras.layers.Conv2DTranspose,
            )

    def test_leap_cnn_interp(self):
        arch = leap.LeapCNN(
            filters=8,
            filters_rate=2,
            down_blocks=3,
            down_convs_per_block=3,
            up_blocks=3,
            up_interpolate=True,
            up_convs_per_block=2,
        )
        x_in = tf.keras.layers.Input((64, 64, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 37)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 30)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 120272)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 120272)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 64, 64, 16))
        with self.subTest("layer instance type by name"):
            self.assertIsInstance(
                model.get_layer("stack0_dec0_s8_to_s4_interp_bilinear"),
                tf.keras.layers.UpSampling2D,
            )

    def test_leap_cnn_reference_from_config(self):
        arch = leap.LeapCNN.from_config(
            LEAPConfig(
                max_stride=8,
                output_stride=1,
                filters=64,
                filters_rate=2,
                up_interpolate=False,
                stacks=1,
            )
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 40)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 36)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 10768896)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 10768896)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 192, 192, 128))
        with self.subTest("number of intermediate features"):
            self.assertEqual(len(x_mid), 3)
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 8)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)
        with self.subTest("layer instance type by name"):
            self.assertIsInstance(
                model.get_layer("stack0_dec0_s8_to_s4_trans_conv"),
                tf.keras.layers.Conv2DTranspose,
            )
