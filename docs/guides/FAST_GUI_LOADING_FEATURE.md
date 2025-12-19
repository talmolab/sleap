# Fast GUI Loading Feature (Deferred)

This document preserves the implementation details for a performance optimization feature that was developed but deferred for separate implementation. The feature addresses slow GUI loading times when working with ImageVideo backends on network filesystems.

## Problem Statement

When opening `.slp` files with ImageVideo backends (image sequences) stored on network filesystems (e.g., SMB/CIFS shares), the SLEAP GUI experiences significant delays:

1. **Root Cause**: The `VideosTableModel.item_to_data()` method accesses `backend.img_shape` to display video metadata (height, width, channels) in the videos list.

2. **Performance Issue**: For ImageVideo backends, accessing `.img_shape` triggers `cv2.imread()` to read the first frame and determine dimensions. On network filesystems, this I/O operation is extremely slow (often 1-5 seconds per video).

3. **User Impact**: With multiple videos, the GUI can freeze for 30+ seconds during initial load.

## Solution Overview

The solution involves using cached shape data from `backend_metadata` (stored in the `.slp` file) instead of triggering `cv2.imread()` calls.

### Key Insight

sleap-io stores video dimensions in `backend_metadata` when saving `.slp` files:
- `backend_metadata["height_"]` - frame height
- `backend_metadata["width_"]` - frame width
- `backend_metadata["channels_"]` - number of channels
- `backend_metadata["filenames"]` - list of frame paths (for frame count)

---

## Implementation Code

### 1. Pre-populate Backend Shape Cache (`lf_labels_utils.py`)

Add this function to `sleap/sleap_io_adaptors/lf_labels_utils.py`:

```python
def _prepopulate_backend_shape_cache(video: Video):
    """Pre-populate the backend's cached shape from backend_metadata.

    For ImageVideo backends, accessing backend.shape triggers cv2.imread() to read
    a frame and determine dimensions. On network filesystems, this is very slow.

    The .slp file stores shape in backend_metadata, so we can use that to
    pre-populate the backend's _cached_shape, avoiding the imread entirely.

    Args:
        video: Video object to optimize.
    """
    if video.backend is None:
        return

    # Only pre-populate if shape is in metadata and not already cached
    if (
        hasattr(video.backend, "_cached_shape")
        and video.backend._cached_shape is None
        and "shape" in video.backend_metadata
        and video.backend_metadata["shape"] is not None
    ):
        video.backend._cached_shape = tuple(video.backend_metadata["shape"])
```

**Call this function**:
- After `load_file()` in `load_labels_video_search()`
- Before and after `replace_filename()` calls in `video_callback`

### 2. Optimized `item_to_data` (`dataviews.py`)

Replace `VideosTableModel.item_to_data()` in `sleap/gui/dataviews.py`:

```python
def item_to_data(self, obj, item: "VideoBackend"):
    data = {}
    # Keep reference to original Video to access backend_metadata
    original_video = item if isinstance(item, Video) else None
    if isinstance(item, Video):
        item = item.backend

    # PERFORMANCE FIX: Avoid accessing img_shape directly as it triggers
    # cv2.imread() for ImageVideo backends, which is very slow on network
    # filesystems. Instead, try to get shape from cached sources.
    img_shape = None
    if item is not None:
        if hasattr(item, "_cached_shape") and item._cached_shape is not None:
            img_shape = item._cached_shape[1:]  # Skip frames dimension
    if img_shape is None and original_video is not None:
        meta = original_video.backend_metadata
        if "height_" in meta and "width_" in meta:
            h = meta.get("height_")
            w = meta.get("width_")
            c = meta.get("channels_", 1)
            if h is not None and w is not None:
                img_shape = (h, w, c)

    for property in self.properties:
        if property == "name":
            if item is None:
                data[property] = "N/A"
            else:
                data[property] = (
                    Path(item.filename).name
                    if isinstance(item.filename, str)
                    else item.filename[0]
                )
        elif property == "filepath":
            if item is None:
                data[property] = "N/A"
            else:
                data[property] = (
                    str(Path(item.filename).parent)
                    if isinstance(item.filename, str)
                    else item.filename[0]
                )
        elif property == "height":
            data[property] = img_shape[0] if img_shape else "N/A"
        elif property == "width":
            data[property] = img_shape[1] if img_shape else "N/A"
        elif property == "channels":
            data[property] = (
                img_shape[2] if img_shape and len(img_shape) > 2 else "N/A"
            )
        elif property == "frames":
            frames_val = None
            if item is not None:
                has_cached = hasattr(item, "_cached_shape")
                if has_cached and item._cached_shape is not None:
                    frames_val = item._cached_shape[0]
                elif hasattr(item, "filename") and isinstance(item.filename, list):
                    frames_val = len(item.filename)
            if frames_val is None and original_video is not None:
                meta = original_video.backend_metadata
                if "filenames" in meta and meta["filenames"]:
                    frames_val = len(meta["filenames"])
            data[property] = frames_val if frames_val is not None else "N/A"
        else:
            data[property] = getattr(item, property) if item else "N/A"
    return data
```

### 3. Video Worker Optimization (`video_worker.py`)

Add shape cache pre-population in `FrameLoaderThread`:

```python
def _prepopulate_shape_cache(self, video: sio.Video):
    """Pre-populate backend shape cache from metadata to avoid slow imread."""
    if video.backend is None:
        return
    if (
        hasattr(video.backend, "_cached_shape")
        and video.backend._cached_shape is None
        and "shape" in video.backend_metadata
        and video.backend_metadata["shape"] is not None
    ):
        video.backend._cached_shape = tuple(video.backend_metadata["shape"])

def request_frame(self, video: sio.Video, frame_idx: int):
    """Request loading a frame from a video."""
    # Pre-populate shape cache before any operations
    self._prepopulate_shape_cache(video)

    if not video.is_open:
        video.open_backend = True
        video.open()

    # ... rest of implementation
```

### 4. Training Dialog Fix (`app.py`)

Add helper to avoid triggering video shape access in `_get_frames_for_prediction()`:

```python
def get_video_frame_count(video) -> int:
    """Get frame count without triggering slow imread operations.

    For ImageVideo backends on network filesystems, accessing video.shape[0]
    triggers cv2.imread() which is very slow. This helper uses cached
    metadata when available.
    """
    # Try cached shape first
    if video.backend is not None:
        if hasattr(video.backend, "_cached_shape") and video.backend._cached_shape is not None:
            return video.backend._cached_shape[0]

    # Try backend_metadata
    meta = video.backend_metadata
    if "filenames" in meta and meta["filenames"]:
        return len(meta["filenames"])

    # Fallback to len(video) - may be slow but necessary
    return len(video)
```

Then use `get_video_frame_count(video)` instead of `len(video)` or `video.shape[0]` in frame prediction logic.

---

## Test Coverage

The following tests should be created in `tests/test_fast_gui_loading.py`:

```python
"""Tests for fast GUI loading performance optimizations."""

from pathlib import Path
from unittest.mock import MagicMock
import sleap_io as sio
from sleap_io import Video


class TestPrepopulateBackendShapeCache:
    """Tests for the _prepopulate_backend_shape_cache function."""

    def test_prepopulate_with_valid_metadata(self):
        """Test that shape cache is populated from backend_metadata."""
        from sleap.sleap_io_adaptors.lf_labels_utils import (
            _prepopulate_backend_shape_cache,
        )

        video = MagicMock(spec=Video)
        video.backend = MagicMock()
        video.backend._cached_shape = None
        video.backend_metadata = {"shape": [10, 480, 640, 3]}

        _prepopulate_backend_shape_cache(video)

        assert video.backend._cached_shape == (10, 480, 640, 3)

    def test_prepopulate_with_none_backend(self):
        """Test that function handles None backend gracefully."""
        from sleap.sleap_io_adaptors.lf_labels_utils import (
            _prepopulate_backend_shape_cache,
        )

        video = MagicMock(spec=Video)
        video.backend = None

        # Should not raise
        _prepopulate_backend_shape_cache(video)

    def test_prepopulate_with_already_cached_shape(self):
        """Test that function doesn't overwrite existing cached shape."""
        from sleap.sleap_io_adaptors.lf_labels_utils import (
            _prepopulate_backend_shape_cache,
        )

        video = MagicMock(spec=Video)
        video.backend = MagicMock()
        video.backend._cached_shape = (5, 100, 100, 1)  # Already cached
        video.backend_metadata = {"shape": [10, 480, 640, 3]}

        _prepopulate_backend_shape_cache(video)

        # Should not be overwritten
        assert video.backend._cached_shape == (5, 100, 100, 1)


class TestVideosTableModelItemToData:
    """Tests for VideosTableModel.item_to_data performance fixes."""

    def test_item_to_data_with_cached_shape(self, qtbot):
        """Test that _cached_shape is used instead of img_shape."""
        from sleap.gui.dataviews import VideosTableModel

        mock_video = MagicMock(spec=Video)
        mock_backend = MagicMock()
        mock_backend._cached_shape = (10, 480, 640, 3)
        mock_backend.filename = "test.mp4"
        mock_video.backend = mock_backend
        mock_video.backend_metadata = {}

        model = VideosTableModel(items=[mock_video])
        data = model.item_to_data(mock_video, mock_video)

        assert data["height"] == 480
        assert data["width"] == 640
        assert data["channels"] == 3
        assert data["frames"] == 10

    def test_item_to_data_with_backend_metadata(self, qtbot):
        """Test fallback to backend_metadata for dimensions."""
        from sleap.gui.dataviews import VideosTableModel

        mock_video = MagicMock(spec=Video)
        mock_backend = MagicMock()
        mock_backend._cached_shape = None
        mock_backend.filename = "test.mp4"
        mock_video.backend = mock_backend
        mock_video.backend_metadata = {
            "height_": 720,
            "width_": 1280,
            "channels_": 3,
            "filenames": ["frame1.jpg"] * 20,
        }

        model = VideosTableModel(items=[mock_video])
        data = model.item_to_data(mock_video, mock_video)

        assert data["height"] == 720
        assert data["width"] == 1280
        assert data["channels"] == 3
        assert data["frames"] == 20
```

---

## Known Issues & Considerations

1. **sleap-io metadata keys**: sleap-io uses `height_`, `width_`, `channels_` (with underscores) not `height`, `width`, `channels`. The `shape` key stores `[frames, height, width, channels]`.

2. **Training dialog stall**: The `_get_frames_for_prediction()` function in `app.py` accesses `len(video)` and `video.shape[0]` for all videos when building frame selection options. This needs the `get_video_frame_count()` helper.

3. **replace_filename() triggers close()**: When replacing video filenames, `replace_filename()` calls `open()` which first calls `close()`. The `close()` method accesses `backend.shape` to save metadata, which triggers imread on the OLD (missing) paths. Pre-populating the cache before `replace_filename()` prevents this.

---

## Timeline

- **2024-12**: Initial implementation developed alongside MissingFilesDialog fixes
- **2024-12**: Feature separated for independent testing
- **Status**: Ready for implementation as separate PR

---

## Re-implementation Steps

To re-implement this feature:

1. Add `_prepopulate_backend_shape_cache()` to `lf_labels_utils.py`
2. Call it after `load_file()` and around `replace_filename()` calls
3. Replace `item_to_data()` in `dataviews.py` with optimized version
4. Add `_prepopulate_shape_cache()` to `video_worker.py`
5. Add `get_video_frame_count()` helper to `app.py`
6. Create tests in `tests/test_fast_gui_loading.py`
7. Run full test suite and manual testing with network filesystem videos
