"""Tests for the temporary centroid-only models feature flag."""

import pytest

from sleap.gui.learning.features import is_centroid_models_enabled


def test_disabled_by_default(monkeypatch):
    """The flag is off when the env var is unset."""
    monkeypatch.delenv("SLEAP_ENABLE_CENTROID_MODELS", raising=False)
    assert is_centroid_models_enabled() is False
    assert is_centroid_models_enabled(False) is False


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
@pytest.mark.parametrize("transform", [str.lower, str.upper, str.title])
def test_enabled_for_truthy_values(monkeypatch, value, transform):
    """Truthy values enable the flag regardless of case."""
    monkeypatch.setenv("SLEAP_ENABLE_CENTROID_MODELS", transform(value))
    assert is_centroid_models_enabled() is True
    assert is_centroid_models_enabled(False) is True


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_disabled_for_falsy_values(monkeypatch, value):
    """Falsy values keep the flag off."""
    monkeypatch.setenv("SLEAP_ENABLE_CENTROID_MODELS", value)
    assert is_centroid_models_enabled() is False
    assert is_centroid_models_enabled(False) is False


def test_enabled_by_experimental_features_without_env(monkeypatch):
    """The Experimental Features toggle enables the flag without the env var."""
    monkeypatch.delenv("SLEAP_ENABLE_CENTROID_MODELS", raising=False)
    assert is_centroid_models_enabled(experimental_features=True) is True
    assert is_centroid_models_enabled(True) is True


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_experimental_features_overrides_falsy_env(monkeypatch, value):
    """The Experimental Features toggle enables the flag even with a falsy env."""
    monkeypatch.setenv("SLEAP_ENABLE_CENTROID_MODELS", value)
    assert is_centroid_models_enabled(experimental_features=True) is True
