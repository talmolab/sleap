import numpy as np
import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

from sleap.nn.architectures import unet


class UnetTests(tf.test.TestCase):
    def test_unet_reference(self):
        # Reference implementation from the original paper.
        arch = unet.Unet(
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

        self.assertEqual(len(model.layers), 61)
        self.assertEqual(len(model.trainable_weights), 60)
        self.assertEqual(np.sum(param_counts), 34515968)
        self.assertEqual(model.count_params(), 34519808)
        self.assertAllEqual(model.output.shape, (None, 192, 192, 64))
        self.assertEqual(len(x_mid), 4)
        self.assertEqual(arch.encoder_features_stride, 16)
        self.assertEqual(arch.decoder_features_stride, 1)

    def test_unet_no_middle_block(self):
        arch = unet.Unet(
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

        self.assertEqual(len(model.layers), 33)
        self.assertEqual(len(model.trainable_weights), 32)
        self.assertEqual(np.sum(param_counts), 32608)
        self.assertEqual(model.count_params(), 32704)
        self.assertAllEqual(model.output.shape, (None, 192, 192, 8))
        self.assertEqual(len(x_mid), 2)
        self.assertEqual(arch.encoder_features_stride, 4)
        self.assertEqual(arch.decoder_features_stride, 1)

    def test_stacked_unet(self):
        arch = unet.Unet(
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

        self.assertEqual(len(model.layers), 208)
        self.assertEqual(len(model.trainable_weights), 192)
        self.assertEqual(np.sum(param_counts), 23596560)
        self.assertEqual(model.count_params(), 23602512)
        self.assertAllEqual(
            [out.shape for out in model.output],
            [(None, 160, 160, 16)] * 3)
        self.assertEqual([len(mid) for mid in x_mid], [5, 5, 5])
        self.assertEqual(arch.encoder_features_stride, 32)
        self.assertEqual(arch.decoder_features_stride, 1)

    def test_stacked_unet_with_stem(self):
        arch = unet.Unet(
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

        self.assertEqual(len(model.layers), 140)
        self.assertEqual(len(model.trainable_weights), 128)
        self.assertEqual(np.sum(param_counts), 23457264)
        self.assertEqual(model.count_params(), 23462640)
        self.assertAllEqual(
            [out.shape for out in model.output],
            [(None, 40, 40, 64)] * 3)
        self.assertEqual([len(mid) for mid in x_mid], [3, 3, 3])
        self.assertEqual(arch.encoder_features_stride, 32)
        self.assertEqual(arch.decoder_features_stride, 4)
