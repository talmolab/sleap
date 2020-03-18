import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

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
            stem_blocks=0,
            down_blocks=4,
            middle_block=True,
            up_blocks=4,
            up_interpolate=False,
            block_contraction=False,
        )
        x_in = tf.keras.layers.Input((192, 192, 1))
        x, x_mid = arch.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        layer_shapes_gt = {
            "stack0_enc0_conv0": (None, 192, 192, 64),
            "stack0_enc0_conv1": (None, 192, 192, 64),
            "stack0_enc1_pool": (None, 96, 96, 64),
            "stack0_enc1_conv0": (None, 96, 96, 128),
            "stack0_enc1_conv1": (None, 96, 96, 128),
            "stack0_enc2_pool": (None, 48, 48, 128),
            "stack0_enc2_conv0": (None, 48, 48, 256),
            "stack0_enc2_conv1": (None, 48, 48, 256),
            "stack0_enc3_pool": (None, 24, 24, 256),
            "stack0_enc3_conv0": (None, 24, 24, 512),
            "stack0_enc3_conv1": (None, 24, 24, 512),
            "stack0_enc4_last_pool": (None, 12, 12, 512),
            "stack0_enc5_middle_expand_conv0": (None, 12, 12, 1024),
            "stack0_enc6_middle_contract_conv0": (None, 12, 12, 1024),
            "stack0_dec0_s16_to_s8_trans_conv": (None, 24, 24, 512),
            "stack0_dec0_s16_to_s8_refine_conv0": (None, 24, 24, 512),
            "stack0_dec0_s16_to_s8_refine_conv1": (None, 24, 24, 512),
            "stack0_dec1_s8_to_s4_trans_conv": (None, 48, 48, 256),
            "stack0_dec1_s8_to_s4_refine_conv0": (None, 48, 48, 256),
            "stack0_dec1_s8_to_s4_refine_conv1": (None, 48, 48, 256),
            "stack0_dec2_s4_to_s2_trans_conv": (None, 96, 96, 128),
            "stack0_dec2_s4_to_s2_refine_conv0": (None, 96, 96, 128),
            "stack0_dec2_s4_to_s2_refine_conv1": (None, 96, 96, 128),
            "stack0_dec3_s2_to_s1_trans_conv": (None, 192, 192, 64),
            "stack0_dec3_s2_to_s1_refine_conv0": (None, 192, 192, 64),
            "stack0_dec3_s2_to_s1_refine_conv1": (None, 192, 192, 64),
        }

        with self.subTest("layer shapes"):
            layer_shapes = {
                l.name: l.output_shape
                for l in model.layers
                if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.MaxPool2D))
            }
            self.assertEqual(layer_shapes, layer_shapes_gt)
        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 53)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 44)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 34512128)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 34512128)
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
            self.assertEqual(len(model.layers), 25)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 20)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 16320)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 16320)
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
            self.assertEqual(len(model.layers), 178)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 132)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23590608)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23590608)
        with self.subTest("output shape"):
            self.assertAllEqual(
                [out.shape for out in model.output], [(None, 160, 160, 16)] * 3
            )
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
            self.assertEqual(len(model.layers), 122)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 92)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23396592)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23396592)
        with self.subTest("output shape"):
            self.assertAllEqual(
                [out.shape for out in model.output], [(None, 40, 40, 64)] * 3
            )
        with self.subTest("number of intermediate features"):
            self.assertEqual([len(mid) for mid in x_mid], [3, 3, 3])
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 32)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 4)

    def test_from_config(self):
        arch = unet.UNet.from_config(
            UNetConfig(
                stem_stride=None,
                max_stride=2 ** 4,
                output_stride=1,
                filters=64,
                filters_rate=2,
                middle_block=True,
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
            self.assertEqual(len(model.layers), 53)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 44)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 34512128)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 34512128)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 192, 192, 64))
        with self.subTest("number of intermediate features"):
            self.assertEqual(len(x_mid), 4)
        with self.subTest("encoder stride"):
            self.assertEqual(arch.encoder_features_stride, 16)
        with self.subTest("decoder stride"):
            self.assertEqual(arch.decoder_features_stride, 1)
