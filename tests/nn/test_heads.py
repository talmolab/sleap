import tensorflow as tf

tf.config.experimental.set_visible_devices([], device_type="GPU")  # hide GPUs for test

import json
from jsmin import jsmin
import cattr

from sleap.nn import heads


class HeadsTests(tf.test.TestCase):
    def test_centroid_confmap_num_channels(self):
        self.assertEqual(heads.CentroidConfmap().num_channels, 1)

    def test_centroid_confmap_invalid_anchor_specification(self):
        with self.assertRaisesWithLiteralMatch(ValueError,
            "If using a part anchor, either the anchor_part_name or anchor_part_ind must be set."):
            heads.CentroidConfmap(use_part_anchor=True)

    def test_single_part_confmaps_num_channels(self):
        self.assertEqual(heads.SinglePartConfmaps(
            part_names=["a", "b", "c"]
            ).num_channels, 3)

    def test_multi_part_confmaps_num_channels(self):
        self.assertEqual(heads.MultiPartConfmaps(
            part_names=["a", "b", "c"]
            ).num_channels, 3)

    def test_part_affinity_fields_num_channels(self):
        paf_head = heads.PartAffinityFields(
            edge_inds=[(0, 1), (1, 2), (0, 2)],
            part_names=["a", "b", "c"])
        self.assertTrue(paf_head.num_channels, 6)

    def test_part_affinity_fields_underdetermined_part_names(self):
        with self.assertRaisesWithLiteralMatch(ValueError,
            "Fewer part names specified (2) than expected from edge indices (3). "
            "Check the edge indices for the part affinity field output head."):
            paf_head = heads.PartAffinityFields(
                edge_inds=[(0, 1), (1, 2), (0, 2)],
                part_names=["a", "b"])
        

    def test_output_head_invalid_type(self):
        with self.assertRaises(ValueError):
            output_head = heads.OutputHead(type="bad_type", config=None, stride=1)


    def test_output_head_from_cattr(self):
        json_dicts = json.loads(jsmin(
            """
            {
                "type": "MultiPartConfmaps",
                "config": {
                    "part_names": ["a", "b", "c"],
                    "sigma": 1.0
                },
                "stride": 4
            }
            """
            ))

        output_head = cattr.structure(json_dicts, heads.OutputHead)

        with self.subTest("class type"):
            self.assertIsInstance(output_head, heads.OutputHead)
        with self.subTest("class type name"):
            self.assertEqual(output_head.type, "MultiPartConfmaps")
        with self.subTest("config type"):
            self.assertIsInstance(output_head.config, heads.MultiPartConfmaps)
        with self.subTest("config part names"):
            self.assertEqual(output_head.config.part_names, ["a", "b", "c"])
        with self.subTest("config sigma"):
            self.assertEqual(output_head.config.sigma, 1.0)
        with self.subTest("stride"):
            self.assertEqual(output_head.stride, 4)
        with self.subTest("num_chanels"):
            self.assertEqual(output_head.num_channels, 3)

    def test_output_head_from_cattr_no_config(self):
        json_dicts = json.loads(jsmin(
            """
            {
                "type": "MultiPartConfmaps",
                "stride": 4
            }
            """
            ))

        with self.assertRaisesWithLiteralMatch(TypeError,
            "__init__() missing 1 required positional argument: 'part_names'"):
            output_head = cattr.structure(json_dicts, heads.OutputHead)

    def test_output_head_from_cattr_invalid_type(self):
        json_dicts = json.loads(jsmin(
            """
            {
                "type": "invalid_head_type"
            }
            """
            ))

        with self.assertRaisesWithLiteralMatch(ValueError,
            "Could not find output type with name: 'invalid_head_type'"):
            output_head = cattr.structure(json_dicts, heads.OutputHead)

    def test_output_head_make_head(self):
        head = heads.OutputHead(type="CentroidConfmap", config=heads.CentroidConfmap(), stride=1)
        x_in = tf.keras.layers.Input((32, 32, 3))
        x = head.make_head(x_in)

        with self.subTest("shape"):
            self.assertAllEqual(x.shape, (None, 32, 32, 1))
        with self.subTest("name"):
            self.assertEqual(x.name.split("/")[0], "CentroidConfmap")
