"""Tests for inference/training subprocess runners."""

import sys

from unittest.mock import MagicMock, patch

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


def test_inference_worker_drains_output_from_fast_exiting_subprocess(qtbot, tmp_path):
    """All subprocess output must be read, even if the process exits instantly.

    Regression test: the read loop used to be gated on ``while proc.poll() is
    None:``, checked *before* each read. ``poll()`` only reports whether the
    process has exited -- it says nothing about whether stdout has been fully
    drained. For a subprocess that finishes fast enough, the loop's next
    ``poll()`` check would already see it as exited and stop reading, silently
    dropping whatever output was still buffered in the pipe. Reproduced
    deterministically (10/10 trials) with a subprocess that prints 20 lines and
    exits immediately: only 1 line was ever captured. Fixed by looping until
    ``readline()`` itself hits true EOF instead of relying on ``poll()``.
    """
    n_lines = 20
    script = (
        f"import sys; [print(f'log line {{i}}') for i in range({n_lines})]; sys.exit(0)"
    )
    output_path = tmp_path / "out.slp"

    task = MagicMock()
    task.make_predict_cli_call.return_value = (
        [sys.executable, "-c", script],
        str(output_path),
    )

    items = MagicMock()
    items.total_frame_count = 0

    for _ in range(10):
        # Each successful run tries to load the (nonexistent) output .slp;
        # patch sio.load_slp so the test doesn't need a real prediction file.
        import sleap.gui.learning.runners as runners_mod

        with patch.object(
            runners_mod.sio, "load_slp", return_value=MagicMock(labeled_frames=[])
        ):
            worker = InferenceWorker(task, items)
            logged = []
            worker.logOutput.connect(logged.append)
            out_path, ret = worker._run_inference_item(MagicMock(), 0, 1)

        assert ret == "success"
        captured = [line for line in logged if line.startswith("log line")]
        assert len(captured) == n_lines, (
            f"expected all {n_lines} lines, only captured {len(captured)}: {captured}"
        )


def test_inference_worker_strips_local_rank_from_subprocess_env(
    qtbot, tmp_path, monkeypatch
):
    """A leftover LOCAL_RANK in the environment must not reach the subprocess.

    sleap-nn's logger only emits INFO-level messages when LOCAL_RANK is 0 or
    unset (a distributed-training concept -- only one worker should log).
    Inference is never a multi-rank job, but if LOCAL_RANK is set to
    something else in the shell (e.g. left over from an unrelated
    torchrun/accelerate invocation earlier in the same session), the
    inference subprocess would silently drop all of its own log output while
    the JSON progress lines (plain prints, not routed through the logger)
    keep working -- exactly a "progress bar works, no logs show up" report.
    """
    monkeypatch.setenv("LOCAL_RANK", "3")

    # Child process reports back exactly what LOCAL_RANK it actually sees.
    script = (
        "import os; "
        "print('LOCAL_RANK seen by subprocess: ' "
        "+ os.environ.get('LOCAL_RANK', '<unset>'))"
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

    with patch("sleap.gui.learning.runners.sio") as mock_sio:
        mock_sio.load_slp.return_value = MagicMock(labeled_frames=[])
        worker._run_inference_item(MagicMock(), 0, 1)

    assert any("LOCAL_RANK seen by subprocess: <unset>" in line for line in logged), (
        logged
    )
