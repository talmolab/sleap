import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import encoder_decoder


class EncoderDecoderTests(tf.test.TestCase):
    def test_simple_conv_block(self):
        block = encoder_decoder.SimpleConvBlock(
            pooling_stride=2,
            num_convs=3,
            filters=16,
            kernel_size=3,
            use_bias=True,
            batch_norm=False,
            batch_norm_before_activation=True,
            activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 2 * 3 + 1)
        self.assertEqual(len(model.trainable_weights), 6)
        self.assertEqual(model.count_params(), 4800)
        self.assertAllEqual(model.output.shape, (None, 4, 4, 16))

    def test_simple_conv_block_bn(self):
        block = encoder_decoder.SimpleConvBlock(
            pooling_stride=2,
            num_convs=3,
            filters=16,
            kernel_size=3,
            use_bias=True,
            batch_norm=True,
            batch_norm_before_activation=True,
            activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 3 * 3 + 1)
        self.assertEqual(len(model.trainable_weights), 12)
        self.assertEqual(model.count_params(), 4992)
        self.assertAllEqual(model.output.shape, (None, 4, 4, 16))
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2D)
        self.assertIsInstance(model.layers[2], tf.keras.layers.BatchNormalization)
        self.assertIsInstance(model.layers[3], tf.keras.layers.Activation)

    def test_simple_conv_block_bn_post(self):
        block = encoder_decoder.SimpleConvBlock(
            pooling_stride=2,
            num_convs=3,
            filters=16,
            kernel_size=3,
            use_bias=True,
            batch_norm=True,
            batch_norm_before_activation=False,
            activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 3 * 3 + 1)
        self.assertEqual(len(model.trainable_weights), 12)
        self.assertEqual(model.count_params(), 4992)
        self.assertAllEqual(model.output.shape, (None, 4, 4, 16))
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2D)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Activation)
        self.assertIsInstance(model.layers[3], tf.keras.layers.BatchNormalization)

    def test_simple_conv_block_no_pool(self):
        block = encoder_decoder.SimpleConvBlock(
            pool=False,
            pooling_stride=2,
            num_convs=3,
            filters=16,
            kernel_size=3,
            use_bias=True,
            batch_norm=True,
            batch_norm_before_activation=True,
            activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 3 * 3)
        self.assertEqual(len(model.trainable_weights), 12)
        self.assertEqual(model.count_params(), 4992)
        self.assertAllEqual(model.output.shape, (None, 8, 8, 16))

    def test_simple_conv_block_pool_before_convs(self):
        block = encoder_decoder.SimpleConvBlock(
            pool=True,
            pool_before_convs=True,
            pooling_stride=2,
            num_convs=3,
            filters=16,
            kernel_size=3,
            use_bias=True,
            batch_norm=True,
            batch_norm_before_activation=True,
            activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 3 * 3 + 1)
        self.assertEqual(len(model.trainable_weights), 12)
        self.assertEqual(model.count_params(), 4992)
        self.assertAllEqual(model.output.shape, (None, 4, 4, 16))
        self.assertIsInstance(model.layers[1], tf.keras.layers.MaxPooling2D)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Conv2D)
        self.assertIsInstance(model.layers[3], tf.keras.layers.BatchNormalization)
        self.assertIsInstance(model.layers[4], tf.keras.layers.Activation)

    def test_simple_upsampling_block(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 2)
        self.assertEqual(len(model.trainable_weights), 0)
        self.assertEqual(model.count_params(), 0)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 1))
        self.assertIsInstance(model.layers[1], tf.keras.layers.UpSampling2D)

    def test_simple_upsampling_block_trans_conv(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=True,
            transposed_conv_filters=8,
            transposed_conv_kernel_size=3,
            transposed_conv_use_bias=True,
            transposed_conv_batch_norm=True,
            transposed_conv_batch_norm_before_activation=True,
            transposed_conv_activation="relu",
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 3)
        self.assertEqual(len(model.trainable_weights), 4)
        self.assertEqual(model.count_params(), 112)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 8))
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2DTranspose)
        self.assertIsInstance(model.layers[2], tf.keras.layers.BatchNormalization)
        self.assertIsInstance(model.layers[3], tf.keras.layers.Activation)

    def test_simple_upsampling_block_trans_conv_bn_post(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=True,
            transposed_conv_filters=8,
            transposed_conv_kernel_size=3,
            transposed_conv_use_bias=True,
            transposed_conv_batch_norm=True,
            transposed_conv_batch_norm_before_activation=False,
            transposed_conv_activation="relu",
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 1 + 3)
        self.assertEqual(len(model.trainable_weights), 4)
        self.assertEqual(model.count_params(), 112)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 8))
        self.assertIsInstance(model.layers[1], tf.keras.layers.Conv2DTranspose)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Activation)
        self.assertIsInstance(model.layers[3], tf.keras.layers.BatchNormalization)

    def test_simple_upsampling_block_ignore_skip_source(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            skip_connection=False,
            skip_add=False,
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        skip_src = tf.keras.Input((16, 16, 1))
        x = block.make_block(x_in, skip_source=skip_src)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 2)
        self.assertEqual(len(model.trainable_weights), 0)
        self.assertEqual(model.count_params(), 0)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 1))
        self.assertIsInstance(model.layers[1], tf.keras.layers.UpSampling2D)

    def test_simple_upsampling_block_skip_add(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            skip_connection=True,
            skip_add=True,
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        skip_src = tf.ones((1, 16, 16, 1))
        x = block.make_block(x_in, skip_source=skip_src)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 3)
        self.assertEqual(len(model.trainable_weights), 0)
        self.assertEqual(model.count_params(), 0)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 1))
        self.assertIsInstance(model.layers[1], tf.keras.layers.UpSampling2D)
        self.assertTrue("add" in model.layers[2].name.lower())  # tf_op_layer_AddV2
        self.assertAllClose(model(tf.ones((1, 8, 8, 1))), tf.ones((1, 16, 16, 1)) * 2)

    def test_simple_upsampling_block_skip_add_adjust_channels(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            skip_connection=True,
            skip_add=True,
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        skip_src = tf.keras.Input((16, 16, 4))
        x = block.make_block(x_in, skip_source=skip_src)
        model = tf.keras.Model([x_in, skip_src], x)

        self.assertEqual(len(model.layers), 5)
        self.assertEqual(len(model.trainable_weights), 2)
        self.assertEqual(model.count_params(), 1 + 4)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 1))
        self.assertIsInstance(model.layers[3], tf.keras.layers.UpSampling2D)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Conv2D)
        self.assertTrue("add" in model.layers[4].name)

    def test_simple_upsampling_block_skip_concat(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            skip_connection=True,
            skip_add=False,
            refine_convs=0,
        )
        x_in = tf.keras.Input((8, 8, 1))
        skip_src = tf.keras.Input((16, 16, 4))
        x = block.make_block(x_in, skip_source=skip_src)
        model = tf.keras.Model([x_in, skip_src], x)

        self.assertEqual(len(model.layers), 4)
        self.assertEqual(len(model.trainable_weights), 0)
        self.assertEqual(model.count_params(), 0)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 5))
        self.assertIsInstance(model.layers[2], tf.keras.layers.UpSampling2D)
        self.assertIsInstance(model.layers[3], tf.keras.layers.Concatenate)

    def test_simple_upsampling_block_refine_convs(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            skip_connection=True,
            refine_convs=2,
            refine_convs_filters=16,
            refine_convs_use_bias=True,
            refine_convs_kernel_size=3,
            refine_convs_batch_norm=True,
            refine_convs_batch_norm_before_activation=True,
            refine_convs_activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 8)
        self.assertEqual(len(model.trainable_weights), 8)
        self.assertEqual(model.count_params(), 2608)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 16))
        self.assertIsInstance(model.layers[1], tf.keras.layers.UpSampling2D)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Conv2D)
        self.assertIsInstance(model.layers[3], tf.keras.layers.BatchNormalization)
        self.assertIsInstance(model.layers[4], tf.keras.layers.Activation)

    def test_simple_upsampling_block_refine_convs_bn_post(self):
        block = encoder_decoder.SimpleUpsamplingBlock(
            upsampling_stride=2,
            transposed_conv=False,
            interp_method="bilinear",
            skip_connection=True,
            refine_convs=2,
            refine_convs_filters=16,
            refine_convs_use_bias=True,
            refine_convs_kernel_size=3,
            refine_convs_batch_norm=True,
            refine_convs_batch_norm_before_activation=False,
            refine_convs_activation="relu",
        )
        x_in = tf.keras.Input((8, 8, 1))
        x = block.make_block(x_in)
        model = tf.keras.Model(x_in, x)

        self.assertEqual(len(model.layers), 8)
        self.assertEqual(len(model.trainable_weights), 8)
        self.assertEqual(model.count_params(), 2608)
        self.assertAllEqual(model.output.shape, (None, 16, 16, 16))
        self.assertIsInstance(model.layers[1], tf.keras.layers.UpSampling2D)
        self.assertIsInstance(model.layers[2], tf.keras.layers.Conv2D)
        self.assertIsInstance(model.layers[3], tf.keras.layers.Activation)
        self.assertIsInstance(model.layers[4], tf.keras.layers.BatchNormalization)
