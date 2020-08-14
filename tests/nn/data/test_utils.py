import numpy as np
import tensorflow as tf
from sleap.nn.system import use_cpu_only; use_cpu_only()  # hide GPUs for test

from sleap.nn.data import utils


def test_ensure_list():
    assert utils.ensure_list([0, 1, 2]) == [0, 1, 2]
    assert utils.ensure_list(0) == [0]
    assert utils.ensure_list([0]) == [0]


def test_expand_to_rank():
    np.testing.assert_array_equal(
        utils.expand_to_rank(tf.range(3), target_rank=2, prepend=True), [[0, 1, 2]]
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(tf.range(3), target_rank=3, prepend=True), [[[0, 1, 2]]]
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(tf.range(3), target_rank=2, prepend=False), [[0], [1], [2]]
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(
            tf.reshape(tf.range(3), [1, 3]), target_rank=2, prepend=True
        ),
        [[0, 1, 2]],
    )
    np.testing.assert_array_equal(
        utils.expand_to_rank(
            tf.reshape(tf.range(2 * 3 * 4), [2, 3, 4]),
            target_rank=2),
        tf.reshape(tf.range(2 * 3 * 4), [2, 3, 4])
    )


def test_make_grid_vector():
    xv, yv = utils.make_grid_vectors(image_height=4, image_width=3, output_stride=1)

    assert xv.dtype == tf.float32
    assert xv.shape == (3,)
    assert yv.dtype == tf.float32
    assert yv.shape == (4,)

    np.testing.assert_allclose(xv, [0, 1, 2])
    np.testing.assert_allclose(yv, [0, 1, 2, 3])

    xv, yv = utils.make_grid_vectors(image_height=4, image_width=3, output_stride=2)
    np.testing.assert_allclose(xv, [0, 2])
    np.testing.assert_allclose(yv, [0, 2])


def test_gaussian_pdf():
    assert utils.gaussian_pdf(0, sigma=1) == 1.0
    assert utils.gaussian_pdf(1, sigma=1) == 0.6065306597126334
    assert utils.gaussian_pdf(1, sigma=2) == 0.8824969025845955


def test_describe_tensors():
    desc = utils.describe_tensors(
        dict(
            tens=tf.ones((1, 2), dtype=tf.uint8),
            rag=tf.ragged.stack([tf.ones((3,)), tf.ones((4,))], axis=0),
            np=np.array([1, 2])
        ), return_description=True,
    )
    assert desc == "\n".join([
        "tens: type=EagerTensor, shape=(1, 2), dtype=tf.uint8, "
        "device=/job:localhost/replica:0/task:0/device:CPU:0",
        " rag: type=RaggedTensor, shape=(2, None), dtype=tf.float32, device=N/A",
        "  np: type=ndarray, shape=(2,), dtype=int32, device=N/A",
        ])

