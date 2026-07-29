"""Tests for inference/training subprocess runners."""

import sys

from unittest.mock import MagicMock

from sleap.gui.learning.runners import InferenceWorker


def test_inference_worker_reads_non_utf8_subprocess_output(qtbot, tmp_path):
    """Reading inference subprocess output must not crash on a stray non-UTF-8 byte.

    Regression test for gh discussion #2744 / sleap-nn #655: on Windows the GUI
    decoded the inference subprocess output using the cp1252 locale default, so a
    byte like ``0x90`` (e.g. from a ``rich``-rendered traceback or progress bar)
    raised ``UnicodeDecodeError: 'charmap' codec can't decode byte 0x90`` and
    masked the real error. A lone ``0x90`` is also invalid UTF-8, so it reproduces
    the crash on any platform when the reader does not pass ``errors="replace"``.
    """
    # Child process emits a lone 0x90 byte (invalid UTF-8 / cp1252-undefined),
    # stays alive briefly so the parent loop reads + decodes it, then exits
    # non-zero so the success/load-slp branch is skipped.
    script = (
        "import sys, time; "
        "sys.stdout.buffer.write(b'progress \\x90 line\\n'); "
        "sys.stdout.flush(); "
        "time.sleep(0.5); "
        "sys.exit(1)"
    )
    output_path = str(tmp_path / "out.slp")

    task = MagicMock()
    task.make_predict_cli_call.return_value = (
        [sys.executable, "-c", script],
        output_path,
    )

    items = MagicMock()
    items.total_frame_count = 0

    worker = InferenceWorker(task, items)

    logged = []
    worker.logOutput.connect(logged.append)

    # Must not raise UnicodeDecodeError while reading the subprocess output.
    out_path, ret = worker._run_inference_item(MagicMock(), 0, 1)

    assert out_path == output_path
    assert ret == 1  # non-zero exit propagated; no decode crash
    # The malformed line was still captured (with the bad byte replaced).
    assert any("progress" in line and "line" in line for line in logged)


def test_inference_worker_surfaces_structured_gui_error(qtbot, tmp_path):
    """A sleap-nn ``--gui`` structured error line must reach the log pane.

    sleap-nn's ``--gui`` mode emits ``{"error": true, "type": ..., "message":
    ...}`` on stdout before re-raising (``_emit_gui_error``/``_run_guarded``),
    specifically so a GUI reader can show a clean message instead of a raw
    traceback. Without handling this shape, it's valid JSON but doesn't match
    the progress-line shape (``n_processed``/``n_total``), so it was silently
    dropped: no progress update, no log line, nothing shown to the user.
    """
    script = (
        "import json, sys; "
        "print(json.dumps({'n_processed': 1, 'n_total': 10, "
        "'rate': 5.0, 'eta': 2.0}), flush=True); "
        "print(json.dumps({'error': True, 'type': 'FileNotFoundError', "
        "'message': 'Model path does not exist: /bogus'}), flush=True); "
        "sys.exit(1)"
    )
    output_path = str(tmp_path / "out.slp")

    task = MagicMock()
    task.make_predict_cli_call.return_value = (
        [sys.executable, "-c", script],
        output_path,
    )

    items = MagicMock()
    items.total_frame_count = 0

    worker = InferenceWorker(task, items)

    logged = []
    progress = []
    worker.logOutput.connect(logged.append)
    worker.progressUpdate.connect(lambda n, total: progress.append((n, total)))

    out_path, ret = worker._run_inference_item(MagicMock(), 0, 1)

    assert ret == 1
    assert progress == [(1, 10)]  # the progress line before the error still works
    assert any(
        "FileNotFoundError" in line and "Model path does not exist" in line
        for line in logged
    )
