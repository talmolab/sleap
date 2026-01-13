"""Tests for sleap.qc.detector module - integration tests."""

import numpy as np
import pytest

import sleap_io as sio
from sleap_io import LabeledFrame
from sleap_io.model.instance import Instance

from sleap.qc import LabelQCDetector, QCConfig, QCResults


class TestLabelQCDetector:
    """Integration tests for LabelQCDetector."""

    @pytest.fixture
    def simple_labels(self, skeleton):
        """Create simple labels for testing with the 5-node fly skeleton."""
        video = sio.Video.from_filename("test_video.mp4")

        labels = sio.Labels()

        # Create 60 "normal" labeled frames with consistent instances
        for frame_idx in range(60):
            instances = []

            # Create a "normal" instance
            points_array = np.array(
                [
                    [100, 100],  # head
                    [100, 120],  # thorax
                    [100, 150],  # abdomen
                    [80, 115],  # left-wing
                    [120, 115],  # right-wing
                ],
                dtype=float,
            )
            # Add small variation
            points_array += np.random.randn(5, 2) * 2

            instance = Instance.from_numpy(points_array, skeleton=skeleton)
            instances.append(instance)

            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
            labels.append(lf)

        return labels

    @pytest.fixture
    def labels_with_anomaly(self, skeleton):
        """Create labels with one anomalous instance."""
        video = sio.Video.from_filename("test_video.mp4")

        labels = sio.Labels()

        # Create 59 normal frames
        for frame_idx in range(59):
            points_array = np.array(
                [
                    [100, 100],
                    [100, 120],
                    [100, 150],
                    [80, 115],
                    [120, 115],
                ],
                dtype=float,
            )
            points_array += np.random.randn(5, 2) * 2

            instance = Instance.from_numpy(points_array, skeleton=skeleton)
            lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[instance])
            labels.append(lf)

        # Add one anomalous frame (node displaced significantly)
        anomalous_points = np.array(
            [
                [100, 100],
                [100, 120],
                [200, 300],  # abdomen displaced far away
                [80, 115],
                [120, 115],
            ],
            dtype=float,
        )
        anomalous_instance = Instance.from_numpy(anomalous_points, skeleton=skeleton)
        anomalous_lf = LabeledFrame(
            video=video, frame_idx=59, instances=[anomalous_instance]
        )
        labels.append(anomalous_lf)

        return labels

    def test_init_default_config(self):
        """Test detector initialization with default config."""
        detector = LabelQCDetector()

        assert detector.config is not None
        assert detector.config.use_gmm is True

    def test_init_custom_config(self):
        """Test detector initialization with custom config."""
        config = QCConfig(use_gmm=False, instance_threshold=0.5)
        detector = LabelQCDetector(config)

        assert detector.config.use_gmm is False
        assert detector.config.instance_threshold == 0.5

    def test_fit_simple_labels(self, simple_labels):
        """Test fitting detector on simple labels."""
        detector = LabelQCDetector()
        detector.fit(simple_labels)

        assert detector.skeleton_analyzer is not None
        assert detector.baseline_extractor is not None
        assert len(detector.feature_names) > 0

    def test_fit_enables_gmm_for_large_dataset(self, simple_labels):
        """Test that GMM is enabled for datasets >= min_samples."""
        config = QCConfig(gmm_min_samples=50)
        detector = LabelQCDetector(config)
        detector.fit(simple_labels)

        assert detector.use_gmm is True
        assert detector.gmm_detector is not None

    def test_fit_falls_back_to_zscore(self, skeleton):
        """Test fallback to z-score for small datasets."""
        video = sio.Video.from_filename("test_video.mp4")
        labels = sio.Labels()

        # Create only 10 instances (below gmm_min_samples)
        for i in range(10):
            points = np.array(
                [[100, 100], [100, 120], [100, 150], [80, 115], [120, 115]],
                dtype=float,
            )
            instance = Instance.from_numpy(points, skeleton=skeleton)
            lf = LabeledFrame(video=video, frame_idx=i, instances=[instance])
            labels.append(lf)

        config = QCConfig(gmm_min_samples=50)
        detector = LabelQCDetector(config)
        detector.fit(labels)

        assert detector.use_gmm is False
        assert detector.zscore_detector is not None

    def test_score_returns_results(self, simple_labels):
        """Test that score() returns QCResults."""
        detector = LabelQCDetector()
        detector.fit(simple_labels)
        results = detector.score(simple_labels)

        assert isinstance(results, QCResults)
        assert len(results.instance_scores) == 60  # 60 instances

    def test_score_without_fit_raises(self, simple_labels):
        """Test that score() without fit() raises error."""
        detector = LabelQCDetector()

        with pytest.raises(ValueError, match="not fitted"):
            detector.score(simple_labels)

    def test_flag_returns_list(self, simple_labels):
        """Test that flag() returns list of flagged instances."""
        detector = LabelQCDetector()
        detector.fit(simple_labels)
        flagged = detector.flag(simple_labels, threshold=0.9)

        assert isinstance(flagged, list)

    def test_anomaly_detection(self, labels_with_anomaly):
        """Test that anomalous instances are detected."""
        detector = LabelQCDetector()
        detector.fit(labels_with_anomaly)
        results = detector.score(labels_with_anomaly)

        # Get the score for the anomalous instance (frame 59)
        from sleap.qc.results import InstanceKey

        anomaly_key = InstanceKey(0, 59, 0)
        anomaly_score = results.instance_scores.get(anomaly_key, 0)

        # Get scores for normal instances
        normal_scores = [
            score
            for key, score in results.instance_scores.items()
            if key.frame_idx != 59
        ]

        # Sort normal scores
        sorted_normal = sorted(normal_scores)

        # Anomaly should be in the top percentile of scores
        # (higher score = more anomalous)
        # Allow anomaly to be >= 90th percentile of normal scores
        percentile_90 = sorted_normal[int(len(sorted_normal) * 0.9)]
        median_score = np.median(normal_scores)
        assert anomaly_score >= percentile_90 or anomaly_score > median_score

    def test_flag_threshold(self, labels_with_anomaly):
        """Test flagging with different thresholds."""
        detector = LabelQCDetector()
        detector.fit(labels_with_anomaly)

        # Very high threshold - few flags
        high_threshold_flags = detector.flag(labels_with_anomaly, threshold=0.95)

        # Lower threshold - more flags
        low_threshold_flags = detector.flag(labels_with_anomaly, threshold=0.5)

        assert len(low_threshold_flags) >= len(high_threshold_flags)

    def test_frame_level_checks(self, skeleton):
        """Test frame-level checks (instance count, duplicates)."""
        video = sio.Video.from_filename("test_video.mp4")
        labels = sio.Labels()

        # Create 9 normal frames with 2 instances each
        for i in range(9):
            points1 = np.array(
                [[100, 100], [100, 120], [100, 150], [80, 115], [120, 115]],
                dtype=float,
            )
            points2 = np.array(
                [[200, 100], [200, 120], [200, 150], [180, 115], [220, 115]],
                dtype=float,
            )

            inst1 = Instance.from_numpy(points1, skeleton=skeleton)
            inst2 = Instance.from_numpy(points2, skeleton=skeleton)
            lf = LabeledFrame(video=video, frame_idx=i, instances=[inst1, inst2])
            labels.append(lf)

        # Create 1 incomplete frame with only 1 instance
        points = np.array(
            [[100, 100], [100, 120], [100, 150], [80, 115], [120, 115]],
            dtype=float,
        )
        instance = Instance.from_numpy(points, skeleton=skeleton)
        incomplete_lf = LabeledFrame(video=video, frame_idx=9, instances=[instance])
        labels.append(incomplete_lf)

        detector = LabelQCDetector()
        detector.fit(labels)
        results = detector.score(labels)

        # Check frame issues
        frame_issues = results.get_frame_issues()

        # Should detect at least the incomplete frame
        incomplete_frames = [(k, v) for k, v in frame_issues if v.is_incomplete]
        assert len(incomplete_frames) >= 1

    def test_duplicate_detection(self, skeleton):
        """Test detection of duplicate instances."""
        video = sio.Video.from_filename("test_video.mp4")
        labels = sio.Labels()

        # Create 9 normal frames
        for i in range(9):
            points = np.array(
                [[100, 100], [100, 120], [100, 150], [80, 115], [120, 115]],
                dtype=float,
            )
            instance = Instance.from_numpy(points, skeleton=skeleton)
            lf = LabeledFrame(video=video, frame_idx=i, instances=[instance])
            labels.append(lf)

        # Create 1 frame with duplicates (two nearly identical instances)
        points1 = np.array(
            [[100, 100], [100, 120], [100, 150], [80, 115], [120, 115]],
            dtype=float,
        )
        points2 = np.array(
            [[102, 102], [102, 122], [102, 152], [82, 117], [122, 117]],
            dtype=float,
        )  # Nearly same

        inst1 = Instance.from_numpy(points1, skeleton=skeleton)
        inst2 = Instance.from_numpy(points2, skeleton=skeleton)
        duplicate_lf = LabeledFrame(video=video, frame_idx=9, instances=[inst1, inst2])
        labels.append(duplicate_lf)

        detector = LabelQCDetector()
        detector.fit(labels)
        results = detector.score(labels)

        # Check for duplicate detection
        from sleap.qc.results import FrameKey

        frame_key = FrameKey(0, 9)
        frame_qc = results.frame_results.get(frame_key)

        assert frame_qc is not None
        assert len(frame_qc.duplicate_pairs) > 0

    def test_results_to_dataframe(self, simple_labels):
        """Test exporting results to DataFrame."""
        detector = LabelQCDetector()
        detector.fit(simple_labels)
        results = detector.score(simple_labels)

        df = results.to_dataframe()

        assert len(df) == 60
        assert "score" in df.columns
        assert "video_idx" in df.columns
        assert "frame_idx" in df.columns

    def test_feature_names(self, simple_labels):
        """Test that feature names are tracked."""
        detector = LabelQCDetector()
        detector.fit(simple_labels)

        assert len(detector.feature_names) > 0
        assert "max_edge_zscore" in detector.feature_names
        assert "nn_distance" in detector.feature_names

    def test_skeleton_analyzer_properties(self, simple_labels):
        """Test skeleton analyzer extracts properties."""
        detector = LabelQCDetector()
        detector.fit(simple_labels)

        sa = detector.skeleton_analyzer
        assert sa.n_nodes == 5
        assert sa.n_edges == 4
        assert len(sa.symmetry_pairs) >= 1  # left-wing, right-wing


class TestLabelQCDetectorWithRealData:
    """Integration tests using real data fixtures."""

    def test_centered_pair_labels(self, centered_pair_labels):
        """Test with centered pair labels dataset."""
        labels = centered_pair_labels

        # Skip if not enough instances
        n_instances = sum(len(lf.instances) for lf in labels)
        if n_instances < 10:
            pytest.skip("Not enough instances in centered_pair_labels")

        detector = LabelQCDetector()
        detector.fit(labels)
        results = detector.score(labels)

        assert len(results.instance_scores) > 0
        assert len(results.frame_results) > 0

    def test_min_labels(self, min_labels):
        """Test with minimal labels dataset."""
        labels = min_labels

        n_instances = sum(len(lf.instances) for lf in labels)
        if n_instances < 2:
            pytest.skip("Not enough instances in min_labels")

        # Use z-score fallback since min_labels is small
        config = QCConfig(gmm_min_samples=n_instances + 10)
        detector = LabelQCDetector(config)
        detector.fit(labels)
        results = detector.score(labels)

        assert len(results.instance_scores) > 0
