from sleap.nn import data


def test_datagen(min_labels):

    import numpy as np
    import math

    training_data = data.TrainingData.from_labels(min_labels)
    ds = training_data.to_ds()

    skeleton = min_labels.skeletons[0]

    conf_data = data.make_confmap_dataset(ds,)

    imgs = []
    confmaps = []
    for img, confmap in conf_data:
        if type(confmap) == tuple:
            confmap = confmap[0]
        imgs.append(img)
        confmaps.append(confmap)

    imgs = np.stack(imgs)
    confmaps = np.stack(confmaps)

    paf_data = data.make_paf_dataset(
        ds, data.SimpleSkeleton.from_skeleton(skeleton).edges,
    )

    pafs = []
    for img, paf in paf_data.take(10):
        pafs.append(paf)

    pafs = np.stack(pafs)

    assert imgs.shape == (1, 384, 384, 1)
    assert math.isclose(np.ptp(imgs / 255), 0.898, abs_tol=0.01)

    assert confmaps.shape == (1, 384, 384, 2)
    assert confmaps.dtype == np.dtype("float32")
    assert math.isclose(np.ptp(confmaps), 0.999, abs_tol=0.01)

    assert pafs.shape == (1, 384, 384, 2)
    assert pafs.dtype == np.dtype("float32")
    assert math.isclose(np.ptp(pafs), 1.57, abs_tol=0.01)
