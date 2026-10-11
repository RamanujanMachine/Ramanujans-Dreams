from dataclasses import dataclass, field
from .configurable import Configurable


@dataclass
class LogConfig(Configurable):
    SLEEP_TO_PRINT: bool = field(
        default=False,
        metadata={"description": "Throttle terminal printing for readability in heavily concurrent output paths."},
    )
    PROFILE: bool = field(
        default=False,
        metadata={"description": "Emit per-timer debug profiling lines."},
    )
    PROFILE_SUMMARY: bool = field(
        default=False,
        metadata={"description": "Accumulate timers and emit a summary section at the end of runs."},
    )
    GENERATE_LOGS: bool = field(
        default=False,
        metadata={"description": "Enable writing structured logs to the configured log file."},
    )
    LOG_FILENAME: str = field(
        default="run.log",
        metadata={"description": "Path to the active debug log file used by the logger backend."},
    )
    EXCEPTION_SHOW_TRACE: bool = field(
        default=False,
        metadata={"description": "Include stack traces for logged exceptions."},
    )
    DEBUG_SHOW_TRACE: bool = field(
        default=False,
        metadata={"description": "Include stack traces for debug-level diagnostic logs."},
    )
    WATCHDOG_ENABLED: bool = field(
        default=True,
        metadata={"description": "Enable background watchdog warnings for procedures that run past their threshold (hang detection)."},
    )
    WATCHDOG_TRAJECTORY_SECONDS: float = field(
        default=300.0,
        metadata={"description": "Seconds after which a still-running per-trajectory Tier-1 compute (walk + LIReC identify + delta) triggers a watchdog warning."},
    )
    WATCHDOG_ATTRIBUTE_SECONDS: float = field(
        default=120.0,
        metadata={"description": "Seconds after which a still-running single trajectory attribute (e.g. asymptotics) triggers a watchdog warning."},
    )


logging_config: LogConfig = LogConfig()
