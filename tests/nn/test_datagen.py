from sleap.nn import data


def test_datagen(min_labels):

    import numpy as np
    import math

    ds = data.pipelines.Pipeline.from_sequence(
        [
            data.providers.LabelsReader(min_labels),
            data.confidence_maps.MultiConfidenceMapGenerator(),
        ]
    ).make_dataset()

    imgs = []
    confmaps = []
    for item in ds:
        imgs.append(item["image"])
        confmaps.append(item["confidence_maps"])

    imgs = np.stack(imgs)
    confmaps = np.stack(confmaps)

    assert imgs.shape == (1, 384, 384, 1)
    assert math.isclose(np.ptp(imgs / 255), 0.898, abs_tol=0.01)

    assert confmaps.shape == (1, 384, 384, 2)
    assert confmaps.dtype == np.dtype("float32")
    assert math.isclose(np.ptp(confmaps), 0.999, abs_tol=0.01)

    skeleton = min_labels.skeletons[0]

    ds = data.pipelines.Pipeline.from_sequence(
        [
            data.providers.LabelsReader(min_labels),
            data.edge_maps.PartAffinityFieldsGenerator(skeletons=skeleton),
        ]
    ).make_dataset()

    pafs = []
    for item in ds:
        pafs.append(item["part_affinity_fields"])

    pafs = np.stack(pafs)

    assert pafs.shape == (1, 384, 384, 1, 2)
    assert pafs.dtype == np.dtype("float32")
