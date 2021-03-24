import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import upsampling
from sleap.nn.config import UpsamplingConfig


class UpsamplingTests(tf.test.TestCase):
    def test_upsampling_stack(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=4,
            upsampling_stride=2,
            transposed_conv=True,
            transposed_conv_batchnorm=True,
            refine_convs=1,
            refine_convs_batchnorm=True,
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((8, 8, 32)), current_stride=16
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertAllEqual(x.shape, (None, 32, 32, 64))
        self.assertEqual(len(intermediate_feats), 3)
        self.assertEqual(intermediate_feats[0].stride, 16)
        self.assertEqual(intermediate_feats[1].stride, 8)
        self.assertEqual(intermediate_feats[2].stride, 4)
        self.assertEqual(len(model.layers), 13)
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2DTranspose)

    def test_upsampling_stack_transposed_filter_rate(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=2,
            upsampling_stride=2,
            transposed_conv=True,
            transposed_conv_filters=16,
            transposed_conv_filters_rate=2,
            refine_convs=0,
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((4, 4, 2)), current_stride=16
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertEqual(model.get_layer("upsample_s16_to_s8_trans_conv").filters, 16)
        self.assertEqual(model.get_layer("upsample_s8_to_s4_trans_conv").filters, 32)
        self.assertEqual(model.get_layer("upsample_s4_to_s2_trans_conv").filters, 64)
        self.assertAllEqual(x.shape, (None, 32, 32, 64))

    def test_upsampling_stack_transposed_filter_rate_shrink(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=2,
            upsampling_stride=2,
            transposed_conv=True,
            transposed_conv_filters=128,
            transposed_conv_filters_rate=0.5,
            refine_convs=0,
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((4, 4, 2)), current_stride=16
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertEqual(model.get_layer("upsample_s16_to_s8_trans_conv").filters, 128)
        self.assertEqual(model.get_layer("upsample_s8_to_s4_trans_conv").filters, 64)
        self.assertEqual(model.get_layer("upsample_s4_to_s2_trans_conv").filters, 32)
        self.assertAllEqual(x.shape, (None, 32, 32, 32))

    def test_upsampling_stack_refine_convs_filter_rate(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=2,
            upsampling_stride=2,
            transposed_conv=False,
            refine_convs=2,
            refine_convs_filters=16,
            refine_convs_filters_rate=2,
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((4, 4, 2)), current_stride=16
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertEqual(model.get_layer("upsample_s16_to_s8_refine0_conv").filters, 16)
        self.assertEqual(model.get_layer("upsample_s16_to_s8_refine1_conv").filters, 16)
        self.assertEqual(model.get_layer("upsample_s8_to_s4_refine0_conv").filters, 32)
        self.assertEqual(model.get_layer("upsample_s8_to_s4_refine1_conv").filters, 32)
        self.assertEqual(model.get_layer("upsample_s4_to_s2_refine0_conv").filters, 64)
        self.assertEqual(model.get_layer("upsample_s4_to_s2_refine1_conv").filters, 64)
        self.assertAllEqual(x.shape, (None, 32, 32, 64))

    def test_upsampling_stack_upsampling_stride4(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=4, upsampling_stride=4
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((8, 8, 32)), current_stride=16
        )

        self.assertAllEqual(x.shape, (None, 32, 32, 64))
        self.assertEqual(len(intermediate_feats), 2)

    def test_upsampling_stack_upsampling_interp(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=8, upsampling_stride=2, transposed_conv=False
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((8, 8, 32)), current_stride=16
        )

        self.assertAllEqual(x.shape, (None, 16, 16, 64))
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)
        self.assertIsInstance(model.layers[1], tf.keras.layers.UpSampling2D)

    def test_upsampling_stack_upsampling_skip(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=2,
            upsampling_stride=2,
            skip_add=False,
            transposed_conv=True,
            transposed_conv_filters=16,
            refine_convs=0,
        )
        skip_sources = [
            upsampling.IntermediateFeature(
                tensor=tf.keras.Input((16, 16, 1)), stride=8
            ),
            upsampling.IntermediateFeature(
                tensor=tf.keras.Input((32, 32, 2)), stride=4
            ),
        ]
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((8, 8, 32)), current_stride=16, skip_sources=skip_sources
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertAllEqual(x.shape, (None, 64, 64, 16))
        self.assertEqual(len(intermediate_feats), 4)
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2DTranspose)
        self.assertIsInstance(model.layers[2], tf.keras.layers.BatchNormalization)
        self.assertIsInstance(model.layers[4], tf.keras.layers.Activation)
        self.assertIsInstance(model.layers[5], tf.keras.layers.Concatenate)
        self.assertAllEqual(model.layers[5].output.shape, (None, 16, 16, 17))

        self.assertIsInstance(model.layers[10], tf.keras.layers.Concatenate)
        self.assertAllEqual(model.layers[10].output.shape, (None, 32, 32, 18))

    def test_upsampling_stack_upsampling_add(self):
        upsampling_stack = upsampling.UpsamplingStack(
            output_stride=2,
            upsampling_stride=2,
            skip_add=True,
            transposed_conv=True,
            transposed_conv_filters=16,
            refine_convs=0,
        )
        skip_sources = [
            upsampling.IntermediateFeature(
                tensor=tf.keras.Input((16, 16, 1)), stride=8
            ),
            upsampling.IntermediateFeature(
                tensor=tf.keras.Input((32, 32, 2)), stride=4
            ),
        ]
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((8, 8, 32)), current_stride=16, skip_sources=skip_sources
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertAllEqual(x.shape, (None, 64, 64, 16))
        self.assertEqual(len(intermediate_feats), 4)
        self.assertAllEqual(
            model.get_layer("upsample_s16_to_s8_skip_conv1x1").output.shape,
            (None, 16, 16, 16),
        )
        self.assertAllEqual(
            model.get_layer("upsample_s8_to_s4_skip_conv1x1").output.shape,
            (None, 32, 32, 16),
        )
        self.assertIsInstance(
            model.get_layer("upsample_s16_to_s8_skip_add"), tf.keras.layers.Add
        )

    def test_upsampling_stack_upsampling_concat(self):
        upsampling_stack = upsampling.UpsamplingStack.from_config(
            UpsamplingConfig(
                method="transposed_conv",
                skip_connections="concatenate",
                block_stride=2,
                filters=64,
                filters_rate=1.0,
                refine_convs=1,
                batch_norm=True,
                transposed_conv_kernel_size=4,
            ),
            output_stride=4,
        )
        x, intermediate_feats = upsampling_stack.make_stack(
            tf.keras.Input((8, 8, 32)), current_stride=16
        )
        model = tf.keras.Model(tf.keras.utils.get_source_inputs(x), x)

        self.assertAllEqual(x.shape, (None, 32, 32, 64))
        self.assertEqual(len(intermediate_feats), 3)
        self.assertEqual(intermediate_feats[0].stride, 16)
        self.assertEqual(intermediate_feats[1].stride, 8)
        self.assertEqual(intermediate_feats[2].stride, 4)
        self.assertEqual(len(model.layers), 13)
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2DTranspose)
