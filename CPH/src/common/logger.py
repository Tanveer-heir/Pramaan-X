"""Structured logging setup with standard logging fallback."""

import logging
import sys

try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    HAS_STRUCTLOG = False

def setup_logger():
    if HAS_STRUCTLOG:
        logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger()
    else:
        # Standard library fallback
        logger_std = logging.getLogger("cph_section2")
        if not logger_std.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ"
            )
            handler.setFormatter(formatter)
            logger_std.addHandler(handler)
            logger_std.setLevel(logging.INFO)

        # Wrap in a simple interface that accepts keyword arguments like structlog
        class FallbackLoggerWrapper:
            def __init__(self, raw_logger):
                self._logger = raw_logger

            def _fmt(self, msg, kwargs):
                if kwargs:
                    extra_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
                    return f"{msg} | {extra_str}"
                return msg

            def info(self, msg, **kwargs):
                self._logger.info(self._fmt(msg, kwargs))

            def warn(self, msg, **kwargs):
                self._logger.warning(self._fmt(msg, kwargs))

            def warning(self, msg, **kwargs):
                self._logger.warning(self._fmt(msg, kwargs))

            def error(self, msg, **kwargs):
                self._logger.error(self._fmt(msg, kwargs))

            def debug(self, msg, **kwargs):
                self._logger.debug(self._fmt(msg, kwargs))

        return FallbackLoggerWrapper(logger_std)

logger = setup_logger()
