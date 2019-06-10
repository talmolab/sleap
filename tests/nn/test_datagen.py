from sleap.nn.datagen import generate_images, generate_confidence_maps, generate_pafs

def test_datagen(min_labels):

    import numpy as np
    import math

    imgs = generate_images(min_labels)

    assert imgs.shape == (1, 384, 384, 1)
    assert imgs.dtype == np.dtype("float32")
    assert math.isclose(np.ptp(imgs), .898, abs_tol=.01)

    confmaps = generate_confidence_maps(min_labels)
    assert confmaps.shape == (1, 384, 384, 2)
    assert confmaps.dtype == np.dtype("float32")
    assert math.isclose(np.ptp(confmaps), .999, abs_tol=.01)


    pafs = generate_pafs(min_labels)
    assert pafs.shape == (1, 384, 384, 2)
    assert pafs.dtype == np.dtype("float32")
    assert math.isclose(np.ptp(pafs), 1.57, abs_tol=.01)