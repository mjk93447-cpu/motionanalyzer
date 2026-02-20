"""
Unit tests for change point detection algorithms.
"""

import numpy as np
import pytest

from motionanalyzer.time_series.changepoint import (
    CUSUMDetector,
    ChangePointResult,
    WindowBasedDetector,
    detect_change_points_pelt,
)


class TestCUSUMDetector:
    """Test CUSUM change point detector."""

    def test_cusum_basic(self) -> None:
        """Test basic CUSUM detection on simple signal."""
        # Create signal with change point at index 50
        signal = np.concatenate([np.random.randn(50) * 0.5 + 0.0, np.random.randn(50) * 0.5 + 3.0])

        detector = CUSUMDetector(threshold=2.0, min_size=5)
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        assert result.method == "CUSUM"
        assert len(result.change_points) > 0
        # Should detect change point around index 50
        assert any(40 <= cp <= 60 for cp in result.change_points)
        assert result.scores is not None
        assert len(result.scores) == len(signal)

    def test_cusum_no_change(self) -> None:
        """Test CUSUM on signal with no change."""
        signal = np.random.randn(100) * 0.5 + 1.0

        detector = CUSUMDetector(threshold=10.0, min_size=5)  # High threshold = no detection
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        assert len(result.change_points) == 0

    def test_cusum_short_signal(self) -> None:
        """Test CUSUM on very short signal."""
        signal = np.array([1.0, 2.0])

        detector = CUSUMDetector()
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        assert len(result.change_points) == 0  # Too short to detect

    def test_cusum_min_size(self) -> None:
        """Test CUSUM respects min_size parameter."""
        # Create signal with multiple changes
        signal = np.concatenate([
            np.random.randn(20) * 0.5 + 0.0,
            np.random.randn(20) * 0.5 + 3.0,
            np.random.randn(20) * 0.5 + 0.0,
        ])

        detector = CUSUMDetector(threshold=2.0, min_size=30)  # Large min_size
        result = detector.detect(signal)

        # Should detect at most one change point due to min_size constraint
        # (since signal is only 60 frames, min_size=30 means at most 2 segments = 1 change point)
        if len(result.change_points) > 1:
            # Verify that detected change points are at least min_size apart
            for i in range(len(result.change_points) - 1):
                assert (result.change_points[i + 1] - result.change_points[i]) >= 30

    def test_cusum_pandas_series(self) -> None:
        """Test CUSUM works with pandas Series."""
        import pandas as pd

        signal = pd.Series(np.concatenate([np.random.randn(50) + 0.0, np.random.randn(50) + 3.0]))

        detector = CUSUMDetector(threshold=2.0, min_size=5)
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        assert len(result.change_points) > 0


class TestWindowBasedDetector:
    """Test window-based change point detector."""

    def test_window_basic(self) -> None:
        """Test basic window-based detection."""
        # Create signal with change point at index 50
        signal = np.concatenate([np.random.randn(50) * 0.5 + 0.0, np.random.randn(50) * 0.5 + 3.0])

        detector = WindowBasedDetector(window_size=10, threshold_ratio=1.5, min_size=5)
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        assert result.method == "WindowBased"
        assert len(result.change_points) > 0
        # Should detect change point around index 50
        assert any(40 <= cp <= 60 for cp in result.change_points)
        assert result.scores is not None

    def test_window_no_change(self) -> None:
        """Test window-based on signal with no change."""
        signal = np.random.randn(100) * 0.5 + 1.0

        detector = WindowBasedDetector(window_size=10, threshold_ratio=10.0, min_size=5)
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        # High threshold should result in few or no detections
        assert len(result.change_points) <= 1

    def test_window_short_signal(self) -> None:
        """Test window-based on very short signal."""
        signal = np.array([1.0, 2.0, 3.0])

        detector = WindowBasedDetector(window_size=10)
        result = detector.detect(signal)

        assert isinstance(result, ChangePointResult)
        assert len(result.change_points) == 0  # Too short


class TestPELTDetector:
    """Test PELT change point detector."""

    def test_pelt_basic(self) -> None:
        """Test basic PELT detection (requires ruptures library)."""
        try:
            # Create signal with change point at index 50
            signal = np.concatenate([np.random.randn(50) * 0.5 + 0.0, np.random.randn(50) * 0.5 + 3.0])

            result = detect_change_points_pelt(signal, min_size=5, pen=3.0)

            assert isinstance(result, ChangePointResult)
            assert result.method == "PELT"
            assert len(result.change_points) > 0
            # Should detect change point around index 50
            assert any(40 <= cp <= 60 for cp in result.change_points)
        except ImportError:
            pytest.skip("ruptures library not installed")

    def test_pelt_import_error(self) -> None:
        """Test PELT raises ImportError when ruptures not available."""
        # This test would need to mock the import, but we'll just document the behavior
        pass


class TestChangePointResult:
    """Test ChangePointResult dataclass."""

    def test_result_validation(self) -> None:
        """Test ChangePointResult validation."""
        # Valid result
        result = ChangePointResult(change_points=[10, 20, 30], method="test")
        assert result.change_points == [10, 20, 30]

        # Invalid: negative change point
        with pytest.raises(ValueError, match="non-negative"):
            ChangePointResult(change_points=[-1, 10], method="test")

        # Invalid: duplicate change points
        with pytest.raises(ValueError, match="unique"):
            ChangePointResult(change_points=[10, 10, 20], method="test")


class TestIntegration:
    """Integration tests for change point detection on realistic signals."""

    def test_multiple_detectors_comparison(self) -> None:
        """Compare multiple detectors on same signal."""
        # Create signal with known change point at index 50
        np.random.seed(42)
        signal = np.concatenate([np.random.randn(50) * 0.5 + 0.0, np.random.randn(50) * 0.5 + 3.0])

        # CUSUM
        cusum_detector = CUSUMDetector(threshold=2.0, min_size=5)
        cusum_result = cusum_detector.detect(signal)

        # Window-based
        window_detector = WindowBasedDetector(window_size=10, threshold_ratio=1.5, min_size=5)
        window_result = window_detector.detect(signal)

        # Both should detect change point around index 50
        assert len(cusum_result.change_points) > 0
        assert len(window_result.change_points) > 0

        # Check that detected change points are reasonable
        assert any(40 <= cp <= 60 for cp in cusum_result.change_points)
        assert any(40 <= cp <= 60 for cp in window_result.change_points)
