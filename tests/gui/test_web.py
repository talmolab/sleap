from sleap.gui.web import get_analytics_data


def test_get_analytics_data():
    analytics_data = get_analytics_data()
    assert "platform" in analytics_data
