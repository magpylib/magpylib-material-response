"""
Centralized logging configuration for magpylib-material-response.

This module follows the loguru library-friendly recipe:

* The package is *disabled* at import time via ``logger.disable(__name__)`` so
  that simply importing :mod:`magpylib_material_response` never modifies the
  user's loguru configuration nor produces any output.
* Calling :func:`configure_logging` enables the package and adds a sink
  *only if the user has not already configured one for this package*.
* :func:`disable_logging` re-disables the package without touching any other
  loguru sinks.

This way the library never calls ``logger.remove()`` and never overrides sinks
configured by the application or by other libraries.
"""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from typing import Any

from loguru import logger

PACKAGE = "magpylib_material_response"

# Default minimum duration (seconds) for `timelog()` to emit a record.
# Updated by `configure_logging(min_log_time=...)` and consulted by
# `magpylib_material_response.utils.timelog` when its own argument is None.
DEFAULT_MIN_LOG_TIME: float = 1.0

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> | "
    "<level>{message}</level>"
)
_NO_TIME_FORMAT = (
    "<level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>"
)

# Track sinks added by this module so we can replace them on reconfigure
# without disturbing sinks configured by the application.
_handler_ids: list[int] = []


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("true", "1", "yes", "on")


def configure_logging(
    level: str | None = None,
    enable_colors: bool | None = None,
    show_time: bool | None = None,
    sink: Any = None,
    min_log_time: float | None = None,
) -> int:
    """
    Enable and configure logging for the package.

    The library is silent by default. Call this function from an application
    to see log messages.

    Parameters
    ----------
    level : str, optional
        Log level. Defaults to ``"INFO"``. Can be overridden with the
        ``MAGPYLIB_LOG_LEVEL`` environment variable.
    enable_colors : bool, optional
        Enable colored output. Defaults to ``True``. Can be overridden with
        ``MAGPYLIB_LOG_COLORS``.
    show_time : bool, optional
        Show timestamps. Defaults to ``True``. Can be overridden with
        ``MAGPYLIB_LOG_TIME``.
    sink : optional
        Loguru sink (file path, stream, or callable). Defaults to
        ``sys.stderr``, following the convention for library logs.
    min_log_time : float, optional
        Default minimum duration (in seconds) for ``timelog()`` blocks to
        emit a record. Steps that complete faster than this are not logged.
        Functions like ``apply_demag`` honour this value when their own
        ``min_log_time`` argument is left at its default. Can be overridden
        with ``MAGPYLIB_LOG_MIN_TIME``. Defaults to ``1.0``.

    Returns
    -------
    int
        The id of the added sink, so callers can remove it with
        ``loguru.logger.remove(handler_id)`` if desired.
    """
    if level is None:
        level = os.getenv("MAGPYLIB_LOG_LEVEL", "INFO")
    if enable_colors is None:
        enable_colors = _env_bool("MAGPYLIB_LOG_COLORS", True)
    if show_time is None:
        show_time = _env_bool("MAGPYLIB_LOG_TIME", True)
    if sink is None:
        sink = sys.stderr
    if min_log_time is None:
        env_val = os.getenv("MAGPYLIB_LOG_MIN_TIME")
        min_log_time = float(env_val) if env_val is not None else 1.0

    global DEFAULT_MIN_LOG_TIME
    DEFAULT_MIN_LOG_TIME = float(min_log_time)

    # Remove only sinks previously added by this module; leave other sinks
    # (configured by the application) untouched.
    for hid in _handler_ids:
        with suppress(ValueError):
            logger.remove(hid)
    _handler_ids.clear()

    fmt = _DEFAULT_FORMAT if show_time else _NO_TIME_FORMAT
    hid = logger.add(
        sink,
        level=level,
        format=fmt,
        colorize=enable_colors,
        filter=PACKAGE,  # only records emitted from this package's modules
    )
    _handler_ids.append(hid)
    logger.enable(PACKAGE)
    return hid


def disable_logging() -> None:
    """
    Disable logging output from the package.

    Removes any sinks previously added by :func:`configure_logging` and
    disables the package via ``logger.disable``. Other sinks configured by the
    application are left untouched.
    """
    for hid in _handler_ids:
        with suppress(ValueError):
            logger.remove(hid)
    _handler_ids.clear()
    logger.disable(PACKAGE)


# Silent by default: do not emit any records from this package until
# `configure_logging()` is called. This does NOT remove user sinks.
logger.disable(PACKAGE)
