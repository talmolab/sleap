"""Tests for sleap.qc.config module."""

from sleap.qc.config import QCConfig


class TestQCConfig:
    """Tests for QCConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QCConfig()

        assert config.use_gmm is True
        assert config.use_curvature == "auto"
        assert config.use_symmetry == "auto"
        assert config.instance_threshold == 0.7
        assert config.gmm_min_samples == 50

    def test_custom_values(self):
        """Test custom configuration values."""
        config = QCConfig(
            use_gmm=False,
            instance_threshold=0.5,
            gmm_min_samples=100,
        )

        assert config.use_gmm is False
        assert config.instance_threshold == 0.5
        assert config.gmm_min_samples == 100

    def test_should_use_curvature_auto(self):
        """Test auto-curvature based on chain length."""
        config = QCConfig(use_curvature="auto")

        # Chain < 5: no curvature
        assert not config.should_use_curvature(3)
        assert not config.should_use_curvature(4)

        # Chain >= 5: use curvature
        assert config.should_use_curvature(5)
        assert config.should_use_curvature(10)

    def test_should_use_curvature_explicit(self):
        """Test explicit curvature setting."""
        config_on = QCConfig(use_curvature=True)
        assert config_on.should_use_curvature(3)  # Regardless of chain length

        config_off = QCConfig(use_curvature=False)
        assert not config_off.should_use_curvature(10)  # Regardless of chain length

    def test_should_use_symmetry_auto(self):
        """Test auto-symmetry based on skeleton properties."""
        config = QCConfig(use_symmetry="auto")

        assert not config.should_use_symmetry(False)
        assert config.should_use_symmetry(True)

    def test_should_use_symmetry_explicit(self):
        """Test explicit symmetry setting."""
        config_on = QCConfig(use_symmetry=True)
        assert config_on.should_use_symmetry(False)

        config_off = QCConfig(use_symmetry=False)
        assert not config_off.should_use_symmetry(True)
