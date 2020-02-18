import sleap

import tensorflow as tf
tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

import json
from jsmin import jsmin
import cattr

from sleap.nn.model import ModelConfig


class ModelConfigTests(tf.test.TestCase):
    def test_model_config_from_cattr(self):
        json_dicts = json.loads(jsmin(
            """
            {
                "architecture": "UNet",
                "backbone": {
                    "filters": 16,
                    "down_blocks": 5,
                    "up_blocks": 3
                },
                "outputs": [
                    {
                        "type": "MultiPartConfmaps",
                        "config": {
                            "sigma": 10.0,
                            "part_names": ["a", "b", "c"]
                        },
                        "stride": 4
                    },
                    {
                        "type": "PartAffinityFields",
                        "config": {
                            "max_distance": 10.0,
                            "edges": [["a", "b"], ["b", "c"], ["a", "c"]]
                        },
                        "stride": 8
                    }
                ]
            }
            """
            ))

        config = cattr.structure(json_dicts, ModelConfig)
        keras_model = config.make_model((160, 160, 1))

        with self.subTest("keras model output shapes"):
            self.assertAllEqual([x.shape for x in keras_model.outputs],
                [(None, 40, 40, 3), (None, 20, 20, 6)])

    def test_model_config_from_legacy_model_cattr(self):
        json_dicts = json.loads(jsmin(
            """
            {
                "model": {
                    "output_type": 0, // CONFIDENCE_MAP
                    "backbone": {
                        "down_blocks": 3,
                        "up_blocks": 3,
                        "convs_per_depth": 2,
                        "num_filters": 16,
                        "kernel_size": 5,
                        "upsampling_layers": true,
                        "interp": "bilinear"
                    },
                    "skeletons": null,
                    "backbone_name": "UNet"
                },
                "trainer": {
                    "scale": 1,
                    "sigma": 5.0,
                    "instance_crop_use_ctr_node": false,
                    "instance_crop_ctr_node_ind": 0
                }
            }
            """
            ))

        skeletons = [sleap.Skeleton.from_names_and_edge_inds(
            node_names=["a", "b", "c"], edge_inds=[[0, 1], [1, 2], [0, 2]])]
        config = ModelConfig.from_legacy_job_cattr(json_dicts, skeletons=skeletons)
        keras_model = config.make_model((160, 160, 1))

        with self.subTest("keras model output shapes"):
            self.assertAllEqual([x.shape for x in keras_model.outputs],
                [(None, 160, 160, 3)])
