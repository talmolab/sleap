import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.architectures import unet
from sleap.nn.config import UNetConfig

class UnetTests(tf.test.TestCase):
    def test_unet_reference(self):
        # Reference implementation from the original paper.
        arch = unet.UNet(
            filters=64,
            filters_rate=2,
            kernel_size=3,
            convs_per_block=2,
            down_blocks=4,
            middle_block=True,
            up_blocks=4,
            up_interpolate=False,
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 61)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 60)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 34515968)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 34519808)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 192, 192, 64))
        with self.subTest("number of intermediate features"):
            self.assertEqual(len(x_mid), 4)
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 16)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)

    def test_unet_no_middle_block(self):
        arch = unet.UNet(
            filters=8,
            filters_rate=2,
            kernel_size=3,
            convs_per_block=2,
            down_blocks=2,
            middle_block=False,
            up_blocks=2,
            up_interpolate=False,
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 33)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 32)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 32608)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 32704)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 192, 192, 8))
        with self.subTest("number of intermediate features"):
            self.assertEqual(len(x_mid), 2)
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 4)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)

    def test_stacked_unet(self):
        arch = unet.UNet(
            stacks=3,
            filters=16,
            filters_rate=2,
            kernel_size=3,
            convs_per_block=2,
            middle_block=True,
            down_blocks=5,
            up_blocks=5,
            up_interpolate=True,
        )
        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 208)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 192)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23596560)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23602512)
        with self.subTest("output shape"):
            self.assertAllEqual(
                [out.shape for out in model.output],
                [(None, 160, 160, 16)] * 3)
        with self.subTest("number of intermediate features"):
            self.assertEqual([len(mid) for mid in x_mid], [5, 5, 5])
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 32)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)

    def test_stacked_unet_with_stem(self):
        arch = unet.UNet(
            stem_blocks=2,
            stacks=3,
            filters=16,
            filters_rate=2,
            kernel_size=3,
            convs_per_block=2,
            middle_block=True,
            down_blocks=3,
            up_blocks=3,
            up_interpolate=True,
        )
        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 144)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 132)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23531120)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23536496)
        with self.subTest("output shape"):
            self.assertAllEqual(
                [out.shape for out in model.output],
                [(None, 40, 40, 64)] * 3)
        with self.subTest("number of intermediate features"):
            self.assertEqual([len(mid) for mid in x_mid], [3, 3, 3])
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 32)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 4)

    def test_from_config(self):
        arch = unet.UNet.from_config(UNetConfig(
            stem_stride=None,
            max_stride=2 ** 4,
            output_stride=1,
            filters=64,
            filters_rate=2,
            middle_block=True,
            up_interpolate=False,
            stacks=1
            ))

        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 61)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 60)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 34515968)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 34519808)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 192, 192, 64))
        with self.subTest("number of intermediate features"):
            self.assertEqual(len(x_mid), 4)
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 16)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)
