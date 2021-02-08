import sleap
import tensorflow as tf
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from sleap.nn.utils import tf_linear_sum_assignment, match_points

sleap.use_cpu_only()


def test_tf_linear_sum_assignment():
    r, c = tf_linear_sum_assignment(tf.cast([[-1, 0], [0, -1]], tf.float32))
    assert_array_equal(r, [0, 1])
    assert_array_equal(c, [0, 1])
    assert r.dtype == tf.int32
    assert c.dtype == tf.int32


def test_match_points():
    inds1, inds2 = match_points([[0, 0], [1, 2]], [[1, 2], [0, 0]])

    assert_array_equal(inds1, [0, 1])
    assert_array_equal(inds2, [1, 0])
