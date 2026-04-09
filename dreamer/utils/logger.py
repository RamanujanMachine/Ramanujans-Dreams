import functools
import time
import inspect
import logging
from enum import Enum, auto
from contextlib import contextmanager
from typing import Callable, Dict, Tuple, Optional
from dreamer.configs.logging import logging_config

# ==========================================
# CONFIGURE PYTHON STANDARD LOGGING BACKEND
# ==========================================

# New definitions
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


# ==========================================
# LOGGER INTERFACE
# ==========================================

class Logger:
    """
    Logging wrapper maintaining the original interface, powered by standard logging.
    """
    timer_mapping: Dict[str, Tuple[int, float]] = dict()
    print_func: Callable = print
    _instance: Optional['Logger'] = None
    _file_handler: Optional[logging.Handler] = None
    _generate_logs_at_import: bool = bool(logging_config.GENERATE_LOGS)
    _last_generate_logs_value: bool = bool(logging_config.GENERATE_LOGS)
    _generate_logs_toggled_since_import: bool = False

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
            cls._generate_logs_toggled_since_import = True
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
        # If GENERATE_LOGS changed after import, append to avoid truncating an existing log history.
        if cls._generate_logs_toggled_since_import:
            return 'a'
        return 'w'

    @classmethod
    def _ensure_file_handler(cls):
        desired_filename = logging_config.LOG_FILENAME

        if cls._file_handler is not None:
            current_filename = getattr(cls._file_handler, 'baseFilename', None)
            if current_filename == desired_filename:
                return
            sys_logger.removeHandler(cls._file_handler)
            cls._file_handler.close()
            cls._file_handler = None

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
            Logger(f'Entered: {func.__name__}', Logger.Levels.debug).log()
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            Logger(f'Exited: {func.__name__} [in {end - start:.6f} seconds]', Logger.Levels.debug).log()
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
    def timer_summary(cls):
        cls._ensure_backend_configured()
        if not logging_config.PROFILE_SUMMARY:
            return
        cls('\n======= profile summary ======', cls.Levels.debug).log()
        for label, (n, s) in cls.timer_mapping.items():
            cls(f"{label}: {n} runs, avg time: {s / n:.6f} seconds", cls.Levels.debug).log()


# ==========================================
# 3. CONFIGURE PYTHON STANDARD LOGGING BACKEND
# ==========================================

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
