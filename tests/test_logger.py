import os

from dreamer.configs.logging import logging_config
from dreamer.utils import logger as logger_mod
from dreamer.utils.logger import Logger


def _remove_all_handlers():
    for handler in list(logger_mod.sys_logger.handlers):
        logger_mod.sys_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def _reset_logger_runtime_state():
    _remove_all_handlers()
    Logger._instance = None
    Logger._file_handler = None
    Logger._last_generate_logs_value = bool(logging_config.GENERATE_LOGS)
    Logger._generate_logs_toggled_since_import = False
    Logger.print_func = print


def _rd_file_handlers_count():
    return sum(1 for h in logger_mod.sys_logger.handlers if getattr(h, "_rd_file_handler", False))


def _rd_console_handlers_count():
    return sum(1 for h in logger_mod.sys_logger.handlers if getattr(h, "_rd_console_handler", False))


def test_logger_is_singleton(tmp_path):
    logging_config.GENERATE_LOGS = False
    logging_config.LOG_FILENAME = str(tmp_path / "singleton.log")
    _reset_logger_runtime_state()

    l1 = Logger("first")
    l2 = Logger("second")

    assert l1 is l2
    assert l2.msg == "second"
    assert _rd_console_handlers_count() == 1


def test_no_file_is_created_when_generate_logs_false(tmp_path):
    log_path = tmp_path / "disabled.log"
    logging_config.GENERATE_LOGS = False
    logging_config.LOG_FILENAME = str(log_path)
    _reset_logger_runtime_state()

    Logger("message while disabled", Logger.Levels.info).log()

    assert not log_path.exists()
    assert Logger._file_handler is None
    assert _rd_file_handlers_count() == 0


def test_file_handler_created_once_when_enabled(tmp_path):
    log_path = tmp_path / "enabled.log"
    logging_config.GENERATE_LOGS = True
    logging_config.LOG_FILENAME = str(log_path)
    _reset_logger_runtime_state()

    Logger("first message", Logger.Levels.info).log()
    first_handler = Logger._file_handler

    Logger("second message", Logger.Levels.info).log()

    assert log_path.exists()
    assert Logger._file_handler is first_handler
    assert _rd_file_handlers_count() == 1


def test_runtime_toggle_from_false_to_true_uses_append_mode(tmp_path):
    log_path = tmp_path / "toggle.log"
    log_path.write_text("seed\n", encoding="utf-8")

    logging_config.GENERATE_LOGS = False
    logging_config.LOG_FILENAME = str(log_path)
    _reset_logger_runtime_state()

    Logger("disabled message", Logger.Levels.info).log()

    logging_config.GENERATE_LOGS = True
    Logger("enabled message", Logger.Levels.info).log()

    content = log_path.read_text(encoding="utf-8")
    assert content.startswith("seed\n")
    assert "enabled message" in content


def test_runtime_toggle_from_true_to_false_removes_file_handler(tmp_path):
    log_path = tmp_path / "toggle_off.log"

    logging_config.GENERATE_LOGS = True
    logging_config.LOG_FILENAME = str(log_path)
    _reset_logger_runtime_state()

    Logger("enabled message", Logger.Levels.info).log()
    assert Logger._file_handler is not None
    assert _rd_file_handlers_count() == 1

    logging_config.GENERATE_LOGS = False
    Logger("now disabled", Logger.Levels.info).log()

    assert Logger._file_handler is None
    assert _rd_file_handlers_count() == 0

    if log_path.exists():
        size_before = os.path.getsize(log_path)
        Logger("still disabled", Logger.Levels.info).log()
        assert os.path.getsize(log_path) == size_before


def test_enabled_from_start_uses_overwrite_mode(tmp_path):
    log_path = tmp_path / "overwrite.log"
    log_path.write_text("old-content\n", encoding="utf-8")

    logging_config.GENERATE_LOGS = True
    logging_config.LOG_FILENAME = str(log_path)
    _reset_logger_runtime_state()

    Logger("fresh-line", Logger.Levels.info).log()

    content = log_path.read_text(encoding="utf-8")
    assert "fresh-line" in content
    assert "old-content" not in content


