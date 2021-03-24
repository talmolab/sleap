import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only

use_cpu_only()  # hide GPUs for test

from sleap.nn.architectures import upsampling
from sleap.nn.architectures import resnet
from sleap.nn.config import ResNetConfig


class ResnetTests(tf.test.TestCase):
    def test_resnet50(self):
        resnet50 = resnet.ResNet50(pretrained=False, features_output_stride=32)
        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet50.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 175)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 212)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23528320)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23581440)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 5, 5, 2048))

    def test_resnet50_stride16(self):
        resnet50 = resnet.ResNet50(pretrained=False, features_output_stride=16)

        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet50.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 175)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 212)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23528320)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23581440)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 10, 10, 2048))
        with self.subTest("feature output stride"):
            self.assertEqual(model.get_layer("conv5_block1_1_conv").strides, (1, 1))
        with self.subTest("feature output dilation rate"):
            self.assertEqual(
                model.get_layer("conv5_block1_1_conv").dilation_rate, (2, 2)
            )

    def test_resnet50_upsampling(self):
        resnet50 = resnet.ResNet50(
            pretrained=False,
            frozen=False,
            features_output_stride=32,
            upsampling_stack=upsampling.UpsamplingStack(
                output_stride=4, refine_convs_filters=64
            ),
        )

        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet50.make_backbone(x_in)

        with self.subTest("output shape"):
            self.assertAllEqual(x.shape, (None, 160 // 4, 160 // 4, 64))

    def test_resnet50_pretrained(self):
        resnet50 = resnet.ResNet50(pretrained=True, features_output_stride=16)

        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet50.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 175 + 2)  # adds preprocessing layers
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 212)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23534592)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23587712)

        with self.subTest("exact pretrained weights"):
            layer = model.get_layer("conv5_block1_1_conv")
            self.assertAllClose(
                layer.kernel[:, :, :3, :3],
                np.array(
                    [
                        [
                            [
                                [0.00946995, 0.00118913, -0.01149864],
                                [-0.01221565, 0.01268964, -0.00900999],
                                [0.05596072, 0.01261912, -0.01511286],
                            ]
                        ]
                    ]
                ),
            )

    def test_resnet50_frozen(self):
        resnet50 = resnet.ResNet50(
            pretrained=True, frozen=True, features_output_stride=16
        )

        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet50.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 0)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 0)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23587712)

    def test_resnet101(self):
        resnet101 = resnet.ResNet101(
            pretrained=False, frozen=False, features_output_stride=16
        )

        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet101.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 345)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 416)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 42546560)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 42651904)

    def test_resnet152(self):
        resnet152 = resnet.ResNet152(
            pretrained=False, frozen=False, features_output_stride=16
        )

        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet152.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 515)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 620)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 58213248)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 58364672)

    def test_resnet50_from_config(self):
        resnet50 = resnet.ResNet50.from_config(
            ResNetConfig(
                version="ResNet50",
                weights="random",
                upsampling=None,
                max_stride=32,
                output_stride=32,
            )
        )
        x_in = tf.keras.layers.Input((160, 160, 1))
        x, x_mid = resnet50.make_backbone(x_in)
        model = tf.keras.Model(x_in, x)
        param_counts = [
            np.prod(train_var.shape) for train_var in model.trainable_weights
        ]

        with self.subTest("number of layers"):
            self.assertEqual(len(model.layers), 175)
        with self.subTest("number of trainable weights"):
            self.assertEqual(len(model.trainable_weights), 212)
        with self.subTest("trainable parameter count"):
            self.assertEqual(np.sum(param_counts), 23528320)
        with self.subTest("total parameter count"):
            self.assertEqual(model.count_params(), 23581440)
        with self.subTest("output shape"):
            self.assertAllEqual(model.output.shape, (None, 5, 5, 2048))
