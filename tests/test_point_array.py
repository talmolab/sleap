import numpy as np
import pytest

from sleap.instance import Point, PredictedPoint, PointArray, PredictedPointArray


@pytest.mark.parametrize(
    "p1",
    [
        Point(0.0, 0.0),
        PredictedPoint(0.0, 0.0, 0.0),
        PointArray(3)[0],
        PredictedPointArray(3)[0],
    ],
)
def test_point(p1):
    """
    Test the Point and PredictedPoint API. This is mainly a safety
    check to make sure numpy record array stuff doesn't change
    """

    # Make sure we are getting Points or PredictedPoints only.
    # This makes sure that PointArray(3)[0] returns a point for
    # example
    assert type(p1) in [PredictedPoint, Point]

    # Check getters and setters
    p1.x = 3.0
    assert p1.x == 3.0

    if type(p1) is PredictedPoint:
        p1.score = 30.0
        assert p1.score == 30.0


def test_constructor():
    p = Point(x=1.0, y=2.0, visible=False, complete=True)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.visible == False
    assert p.complete == True

    p = PredictedPoint(x=1.0, y=2.0, visible=False, complete=True, score=0.3)
    assert p.x == 1.0
    assert p.y == 2.0
    assert p.visible == False
    assert p.complete == True
    assert p.score == 0.3


@pytest.mark.parametrize("parray_cls", [PointArray, PredictedPointArray])
def test_point_array(parray_cls):

    p = parray_cls(5)

    # Make sure length works
    assert len(p) == 5
    assert len(p["x"]) == 5
    assert len(p[["x", "y"]]) == 5

    # Check that single point getitem returns a Point class
    if parray_cls is PredictedPointArray:
        assert type(p[0]) is PredictedPoint
    else:
        assert type(p[0]) is Point

    # Check that slices preserve type as well
    assert type(p[0:4]) is type(p)

    # Check field access
    assert type(p.x) is np.ndarray

    # Check make_default
    d1 = parray_cls.make_default(3)
    d2 = parray_cls.make_default(3)

    # I have to convert from structured to unstructured to get this comparison
    # to work.
    from numpy.lib.recfunctions import structured_to_unstructured

    np.testing.assert_array_equal(
        structured_to_unstructured(d1), structured_to_unstructured(d2)
    )


def test_from_and_to_array():
    p = PointArray(3)

    # Do a round trip conversion
    r = PredictedPointArray.to_array(PredictedPointArray.from_array(p))

    from numpy.lib.recfunctions import structured_to_unstructured

    np.testing.assert_array_equal(
        structured_to_unstructured(p), structured_to_unstructured(r)
    )

    # Make sure conversion uses default score
    r = PredictedPointArray.from_array(p)
    assert r.score[0] == PredictedPointArray.make_default(1)[0].score
