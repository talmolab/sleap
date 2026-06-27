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
