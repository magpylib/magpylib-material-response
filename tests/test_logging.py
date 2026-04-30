"""Tests for the logging configuration module."""

from __future__ import annotations

import io

import pytest
from loguru import logger

from magpylib_material_response import configure_logging, disable_logging
from magpylib_material_response.logging_config import PACKAGE


def _emit_from_package(level: str, message: str, **kwargs) -> None:
    """Emit a log record as if from inside the package (for filter testing)."""
    logger.patch(lambda r: r.update(name=PACKAGE + ".test")).log(
        level, message, **kwargs
    )


@pytest.fixture(autouse=True)
def _reset_logging():
    """Ensure each test starts and ends with the package disabled."""
    disable_logging()
    yield
    disable_logging()


def test_silent_by_default():
    """In the default state, the package is disabled in loguru so that no
    record emitted from within the package will reach any sink."""
    # disable_logging in the autouse fixture leaves the package disabled.
    activation = dict(logger._core.activation_list)
    assert activation.get("magpylib_material_response.") is False


def test_configure_logging_enables_package():
    configure_logging(level="DEBUG", enable_colors=False, sink=io.StringIO())
    activation = dict(logger._core.activation_list)
    assert activation.get("magpylib_material_response.") is True


def test_configure_logging_emits_to_sink():
    sink = io.StringIO()
    configure_logging(level="DEBUG", enable_colors=False, sink=sink)
    _emit_from_package("INFO", "hello {x}", x=42)
    assert "hello 42" in sink.getvalue()


def test_configure_logging_does_not_remove_user_sinks():
    user_sink = io.StringIO()
    user_handler = logger.add(user_sink, level="DEBUG")
    try:
        configure_logging(level="DEBUG", enable_colors=False, sink=io.StringIO())
        logger.info("user message")
        assert "user message" in user_sink.getvalue()
    finally:
        logger.remove(user_handler)


def test_disable_logging_silences_package_only():
    user_sink = io.StringIO()
    user_handler = logger.add(user_sink, level="DEBUG")
    try:
        disable_logging()
        logger.info("still here")
        assert "still here" in user_sink.getvalue()
    finally:
        logger.remove(user_handler)


def test_reconfigure_replaces_only_own_sink():
    sink1 = io.StringIO()
    sink2 = io.StringIO()
    configure_logging(level="DEBUG", enable_colors=False, sink=sink1)
    configure_logging(level="DEBUG", enable_colors=False, sink=sink2)
    _emit_from_package("INFO", "after reconfigure")
    assert "after reconfigure" in sink2.getvalue()
    assert sink1.getvalue() == ""


def test_env_var_level(monkeypatch):
    monkeypatch.setenv("MAGPYLIB_LOG_LEVEL", "WARNING")
    sink = io.StringIO()
    configure_logging(enable_colors=False, sink=sink)
    _emit_from_package("INFO", "should not appear")
    _emit_from_package("WARNING", "should appear")
    out = sink.getvalue()
    assert "should not appear" not in out
    assert "should appear" in out


def test_min_log_time_argument_sets_default():
    """`min_log_time` argument must update the module-level default used by
    `timelog`."""
    from magpylib_material_response import logging_config  # noqa: PLC0415

    configure_logging(
        level="DEBUG", enable_colors=False, sink=io.StringIO(), min_log_time=2.5
    )
    assert logging_config.DEFAULT_MIN_LOG_TIME == 2.5


def test_min_log_time_env_var(monkeypatch):
    monkeypatch.setenv("MAGPYLIB_LOG_MIN_TIME", "0.25")
    from magpylib_material_response import logging_config  # noqa: PLC0415

    configure_logging(level="DEBUG", enable_colors=False, sink=io.StringIO())
    assert logging_config.DEFAULT_MIN_LOG_TIME == 0.25


def test_timelog_uses_default_min_log_time():
    """When called with min_log_time=None, timelog must use the configured
    default and emit a record only if the block exceeds it."""
    from magpylib_material_response.utils import timelog  # noqa: PLC0415

    sink = io.StringIO()
    configure_logging(level="DEBUG", enable_colors=False, sink=sink, min_log_time=0.0)
    with timelog("quick step"):
        pass
    assert "Completed: quick step" in sink.getvalue()
