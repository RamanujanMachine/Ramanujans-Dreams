import functools
import time
import inspect
import logging
import os
import platform
import threading
from enum import Enum, auto
from contextlib import contextmanager
from typing import Callable, Dict, Tuple, Optional
from dreamer.configs.logging import logging_config


MESSAGE_LEVEL = 25
INFORM_LEVEL = 26
logging.addLevelName(MESSAGE_LEVEL, "MESSAGE")


class ColorConsoleFormatter(logging.Formatter):
    """
    Custom formatter to inject ANSI colors into terminal output.
    """
    WHITE = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'

    LEVEL_COLORS = {
        logging.DEBUG: WHITE,
        logging.INFO: GREEN,
        MESSAGE_LEVEL: WHITE,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: RED,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, self.WHITE)

        # Format the prefix exactly like your original interface
        if record.levelno == MESSAGE_LEVEL:
            prefix = ""
        elif record.levelno == logging.DEBUG:
            prefix = "[DEBUG] "
        else:
            prefix = f"[{record.levelname}] "

        message = super().format(record)
        return f"{color}{prefix}{message}{self.WHITE}"


class Logger:
    """
    Logging wrapper maintaining the original interface, powered by standard logging.
    """
    timer_mapping: Dict[str, Tuple[int, float]] = dict()
    print_func: Callable = print
    _instance: Optional['Logger'] = None
    _file_handler: Optional[logging.Handler] = None
    _force_overwrite_next_file_open: bool = True
    _generate_logs_at_import: bool = bool(logging_config.GENERATE_LOGS)
    _last_generate_logs_value: bool = bool(logging_config.GENERATE_LOGS)

    class Levels(Enum):
        debug = auto()
        message = auto()
        info = auto()
        warning = auto()
        exception = auto()
        fatal = auto()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, msg, level: Levels = Levels.message, condition=True):
        self.__class__._ensure_backend_configured()
        calling_frame = inspect.stack()[1]
        self.calling_function_name = calling_frame.function
        self.level = level if isinstance(level, Logger.Levels) else Logger.Levels.info
        self.msg = msg
        self.condition = condition

    @classmethod
    def _track_generate_logs_toggle(cls):
        current_value = bool(logging_config.GENERATE_LOGS)
        if current_value != cls._last_generate_logs_value:
            if current_value:
                # Enabling logs at runtime should continue appending in the current run.
                cls._force_overwrite_next_file_open = False
            cls._last_generate_logs_value = current_value

    @classmethod
    def _get_or_create_console_handler(cls):
        for handler in sys_logger.handlers:
            if getattr(handler, '_rd_console_handler', False):
                return handler

        console_handler = DynamicPrintHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColorConsoleFormatter("%(message)s"))
        setattr(console_handler, '_rd_console_handler', True)
        sys_logger.addHandler(console_handler)
        return console_handler

    @classmethod
    def _build_file_handler_mode(cls) -> str:
        if cls._force_overwrite_next_file_open:
            cls._force_overwrite_next_file_open = False
            return 'w'
        return 'a'

    @staticmethod
    def _rotated_log_filename(base_filename: str, run_idx: int) -> str:
        return f"{base_filename}.{run_idx}"

    @classmethod
    def _next_run_index(cls, base_filename: str) -> int:
        run_idx = 1
        while os.path.exists(cls._rotated_log_filename(base_filename, run_idx)):
            run_idx += 1
        return run_idx

    @classmethod
    def start_run(cls):
        """
        Mark the start of a new System.run lifecycle.

        If file logging is enabled, archive the previous base log file as
        LOG_FILENAME.<n> and start a fresh base file for this run.
        """
        cls._track_generate_logs_toggle()
        cls._get_or_create_console_handler()

        if not logging_config.GENERATE_LOGS:
            cls._remove_file_handler()
            return

        desired_filename = os.path.abspath(logging_config.LOG_FILENAME)
        cls._remove_file_handler()

        if os.path.exists(desired_filename):
            run_idx = cls._next_run_index(desired_filename)
            os.replace(desired_filename, cls._rotated_log_filename(desired_filename, run_idx))

        cls._force_overwrite_next_file_open = True
        cls._ensure_file_handler()
        cls._log_system_status()

    @classmethod
    def _log_system_status(cls):
        """
        Emit a run-start diagnostics block to the debug log with host and configuration details.
        :return: None.
        """
        if not logging_config.GENERATE_LOGS:
            return

        from dreamer.configs import config

        os_name = platform.system() or "Unknown"
        architecture = platform.machine() or platform.architecture()[0]
        cpu_count = os.cpu_count()

        sys_logger.debug("========== SYSTEM STATUS ==========")
        sys_logger.debug(f"cores_available={cpu_count if cpu_count is not None else 'Unknown'}")
        sys_logger.debug(f"os={os_name}")
        sys_logger.debug(f"architecture={architecture}")
        sys_logger.debug("configurations:")

        for section_name, section_cfg in config.iter_sections():
            sys_logger.debug(f"  [{section_name}]")
            for entry_name, entry in section_cfg.export_configurations_with_metadata().items():
                formatted_value = repr(entry["value"])
                description = entry["description"]
                if description:
                    sys_logger.debug(f"    {entry_name} = {formatted_value}  # {description}")
                else:
                    sys_logger.debug(f"    {entry_name} = {formatted_value}")

    @classmethod
    def _ensure_file_handler(cls):
        desired_filename = os.path.abspath(logging_config.LOG_FILENAME)
        log_dir = os.path.dirname(desired_filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        recover_missing_file = False

        if cls._file_handler is not None:
            current_filename = getattr(cls._file_handler, 'baseFilename', None)
            if current_filename is not None:
                current_filename = os.path.abspath(str(current_filename))
            if current_filename == desired_filename and os.path.exists(desired_filename):
                return
            if current_filename == desired_filename and not os.path.exists(desired_filename):
                recover_missing_file = True
            sys_logger.removeHandler(cls._file_handler)
            cls._file_handler.close()
            cls._file_handler = None

        if recover_missing_file:
            # Same run + deleted file => recreate and keep appending semantics.
            cls._force_overwrite_next_file_open = False

        file_handler = logging.FileHandler(desired_filename, mode=cls._build_file_handler_mode())
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(filename)s:%(funcName)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        setattr(file_handler, '_rd_file_handler', True)
        sys_logger.addHandler(file_handler)
        cls._file_handler = file_handler

    @classmethod
    def _remove_file_handler(cls):
        if cls._file_handler is None:
            return
        sys_logger.removeHandler(cls._file_handler)
        cls._file_handler.close()
        cls._file_handler = None

    @classmethod
    def _ensure_backend_configured(cls):
        cls._track_generate_logs_toggle()
        cls._get_or_create_console_handler()

        if logging_config.GENERATE_LOGS:
            cls._ensure_file_handler()
        else:
            cls._remove_file_handler()

    def log(self, msg_prefix='', in_function: bool = False, add_stack_trace: bool = False):
        """
        Log a message. Routes to the appropriate standard logger level.
        """
        self.__class__._ensure_backend_configured()
        if not self.condition:
            return

        final_msg = f"{msg_prefix}{self.msg}"
        if in_function:
            final_msg += f' in {self.calling_function_name}'

        match self.level:
            case Logger.Levels.debug:
                if logging_config.GENERATE_LOGS:
                    sys_logger.debug(final_msg, exc_info=add_stack_trace)
            case Logger.Levels.message:
                sys_logger.log(MESSAGE_LEVEL, final_msg)
            case Logger.Levels.info:
                sys_logger.info(final_msg)
            case Logger.Levels.warning:
                sys_logger.warning(final_msg)
            case Logger.Levels.exception:
                sys_logger.error(final_msg, exc_info=add_stack_trace)
            case Logger.Levels.fatal:
                sys_logger.critical(f"{final_msg} in {self.calling_function_name} -> exiting", exc_info=True)
                exit(1)

    @staticmethod
    def log_exec(func):
        if not callable(func):
            raise Exception('Function is not callable, bad decorator')

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            Logger(f'-> Entered: {func.__name__}', Logger.Levels.debug).log()
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            Logger(f'<- Exited: {func.__name__} [in {end - start:.6f} seconds]', Logger.Levels.debug).log()
            return res

        return wrapper

    @staticmethod
    def buffer_print(total: int, text: str, char: str):
        txt_len = len(text) + 2
        t = int((total - txt_len) / 2)
        if txt_len + 2 * t != total:
            return f'{char * t} {text} {char * (t + 1)}'
        return f'{char * t} {text} {char * t}'

    @classmethod
    @contextmanager
    def simple_timer(cls, label):
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            cls._ensure_backend_configured()
            if logging_config.PROFILE:
                sys_logger.debug(f"TIMER | {label}: {end - start:.6f} seconds")
            if logging_config.PROFILE_SUMMARY:
                n, s = cls.timer_mapping.get(label, (0, 0.0))
                cls.timer_mapping[label] = (n + 1, s + (end - start))

    @classmethod
    @contextmanager
    def watchdog(cls, label, threshold_seconds, *, detail=None, repeat=True):
        """Warn while a wrapped procedure is *still running* past a threshold.

        Unlike :meth:`simple_timer` (which logs only after the block returns),
        this spawns a monitoring-only daemon thread that emits a WARNING the
        moment the wrapped block has been running for ``threshold_seconds`` —
        so a true hang is reported even though the call never returns.  This is
        the intended way to locate a stall (e.g. a stuck trajectory walk /
        LIReC identify): the warning carries ``detail`` so the user knows which
        trajectory / shard to inspect.

        The thread performs no computation (it sleeps on an
        :class:`threading.Event`), so it is consistent with the project's
        "no multithreading for CPU work" directive.

        :param label: Short description of the wrapped procedure.
        :param threshold_seconds: Seconds of runtime after which to warn.  A
            falsy or non-positive value (or ``WATCHDOG_ENABLED=False``) makes
            this a no-op with zero overhead.
        :param detail: Optional extra context appended to the warning.  May be
            a string, or a **callable** returning a string — the callable is
            invoked only if the watchdog actually fires, so building an
            expensive descriptor costs nothing on the happy path.
        :param repeat: When ``True`` (default) re-warn every
            ``threshold_seconds`` while the block is still running; when
            ``False`` warn once and stay silent.
        """
        if (
            not logging_config.WATCHDOG_ENABLED
            or not threshold_seconds
            or threshold_seconds <= 0
        ):
            yield
            return

        done = threading.Event()
        start = time.perf_counter()

        def monitor():
            while not done.wait(threshold_seconds):
                elapsed = time.perf_counter() - start
                try:
                    msg = detail() if callable(detail) else detail
                except Exception as e:  # never let the monitor crash the run
                    msg = f"<detail error: {e}>"
                cls(
                    f"[watchdog] {label} still running after {elapsed:.0f}s"
                    + (f" — {msg}" if msg else ""),
                    cls.Levels.warning,
                ).log()
                if not repeat:
                    break

        threading.Thread(
            target=monitor, daemon=True, name=f"watchdog:{label}"
        ).start()
        try:
            yield
        finally:
            done.set()

    @classmethod
    def timer_summary(cls):
        cls._ensure_backend_configured()
        if not logging_config.PROFILE_SUMMARY:
            return
        cls('\n======= profile summary ======', cls.Levels.debug).log()
        for label, (n, s) in cls.timer_mapping.items():
            cls(f"{label}: {n} runs, avg time: {s / n:.6f} seconds", cls.Levels.debug).log()


class DynamicPrintHandler(logging.Handler):
    """
    A custom logging handler that routes messages through whatever function
    is currently assigned to Logger.print_func (e.g., standard print or tqdm.write).
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            Logger.print_func(msg)
        except Exception:
            self.handleError(record)


# Handle configurations and definitions
sys_logger = logging.getLogger("RD-CMF")
sys_logger.setLevel(logging.DEBUG)
sys_logger.propagate = False
