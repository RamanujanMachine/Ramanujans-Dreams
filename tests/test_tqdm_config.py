from dreamer.utils.logger import Logger
from dreamer.utils.ui.tqdm_config import SmartTQDM


def test_smart_tqdm_logs_progress_with_desc(monkeypatch):
    """
    Verify progress logs include the progress-bar description when provided.

    Assumption: debug progress emits once per update when total=10 (10% step).
    Failure mode trapped: logs that ignore `desc` make multi-progress traces ambiguous.
    """
    captured = []

    def _capture_log(self, msg_prefix='', in_function=False, add_stack_trace=False):
        captured.append(self.msg)

    previous_print_func = Logger.print_func
    monkeypatch.setattr(Logger, 'log', _capture_log)

    progress = SmartTQDM(total=10, desc='Analyzing shards')
    try:
        progress.update(1)
    finally:
        progress.close()

    assert 'Analyzing shards - progress: 1 / 10 (10%)' in captured
    assert Logger.print_func is previous_print_func


def test_smart_tqdm_logs_generic_progress_without_desc(monkeypatch):
    """
    Verify fallback progress logs are still emitted when no description is provided.

    Assumption: fallback label remains stable for callers that do not pass `desc`.
    Failure mode trapped: missing fallback text breaks log parsing in debug sessions.
    """
    captured = []

    def _capture_log(self, msg_prefix='', in_function=False, add_stack_trace=False):
        captured.append(self.msg)

    monkeypatch.setattr(Logger, 'log', _capture_log)

    progress = SmartTQDM(total=10)
    try:
        progress.update(1)
    finally:
        progress.close()

    assert 'SYSTEM PROGRESS - progress: 1 / 10 (10%)' in captured

