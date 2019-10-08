from sleap.info.summary import StatisticSeries


def test_frame_statistics(simple_predictions):
    video = simple_predictions.videos[0]
    stats = StatisticSeries(simple_predictions)

    x = stats.get_point_count_series(video)
    assert len(x) == 2
    assert x[0] == 4
    assert x[1] == 4

    x = stats.get_point_score_series(video, "sum")
    assert len(x) == 2
    assert x[0] == 2.4
    assert x[1] == 5.2

    x = stats.get_point_score_series(video, "min")
    assert len(x) == 2
    assert x[0] == 0.5
    assert x[1] == 1.0

    x = stats.get_instance_score_series(video, "sum")
    assert len(x) == 2
    assert x[0] == 7
    assert x[1] == 9

    x = stats.get_instance_score_series(video, "min")
    assert len(x) == 2
    assert x[0] == 2
    assert x[1] == 3

    x = stats.get_point_displacement_series(video, "mean")
    assert len(x) == 2
    assert x[0] == 0
    assert x[1] == 9.0

    x = stats.get_point_displacement_series(video, "max")
    assert len(x) == 2
    assert len(x) == 2
    assert x[0] == 0
    assert x[1] == 18.0
