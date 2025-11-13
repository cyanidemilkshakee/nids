import logging
import os
import sys
import time
from contextlib import contextmanager

DEFAULT_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def setup_logging(name: str = "ai_nids") -> logging.Logger:
    """Configure and return a module-level logger with consistent format.

    Env vars:
      LOG_LEVEL=INFO|DEBUG|WARNING|ERROR
      LOG_TIME=1 (to include time), otherwise default formatter also includes time
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured
        logger.setLevel(DEFAULT_LEVEL)
        return logger

    logger.setLevel(DEFAULT_LEVEL)
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def log_stage(logger: logging.Logger, stage: str):
    """Context manager to log stage start and duration.

    Example:
      with log_stage(log, "load_data"):
          ...
    """
    start = time.time()
    logger.info(f"-> {stage} | start")
    try:
        yield
        dur = time.time() - start
        logger.info(f"<- {stage} | done | {dur:.2f}s")
    except Exception as e:
        dur = time.time() - start
        logger.exception(f"xx {stage} | failed after {dur:.2f}s | {e}")
        raise


def log_kv(logger: logging.Logger, **kwargs):
    """Log a compact key=value line at INFO."""
    parts = [f"{k}={v}" for k, v in kwargs.items()]
    logger.info(" | ".join(parts))
