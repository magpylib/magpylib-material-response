"""
Centralized logging configuration for magpylib-material-response.

This module provides a proper logging setup that:
- Uses named loggers with package hierarchy
- Allows user configuration through environment variables
- Provides sensible defaults for library usage
- Avoids forcing output to stdout unless explicitly requested
"""

from __future__ import annotations

import os
import sys

from loguru import logger


def get_logger(name: str | None = None):
    """
    Get a named logger for the package.
    
    Parameters
    ----------
    name : str, optional
        Logger name. If None, uses the package root logger.
        
    Returns
    -------
    loguru.Logger
        Configured logger instance
    """
    if name is None:
        name = "magpylib_material_response"
    return logger.bind(module=name)


def configure_logging(
    level: str | None = None,
    enable_colors: bool | None = None,
    show_time: bool | None = None,
    sink=None,
) -> None:
    """
    Configure logging for the package.
    
    This function should be called by users who want to see logging output
    from the library. By default, the library doesn't output logs unless
    explicitly configured.
    
    Parameters
    ----------
    level : str, optional
        Log level. Defaults to INFO. Can be overridden with MAGPYLIB_LOG_LEVEL env var.
    enable_colors : bool, optional
        Enable colored output. Defaults to True for interactive environments.
        Can be overridden with MAGPYLIB_LOG_COLORS env var.
    show_time : bool, optional
        Show timestamps in log messages. Defaults to True.
        Can be overridden with MAGPYLIB_LOG_TIME env var.
    sink : optional
        Log sink. Defaults to sys.stderr. Use sys.stdout for stdout output.
    """
    # Remove existing handlers to avoid duplicates
    logger.remove()
    
    # Get configuration from environment or use defaults
    if level is None:
        level = os.getenv("MAGPYLIB_LOG_LEVEL", "INFO")
    
    if enable_colors is None:
        enable_colors = os.getenv("MAGPYLIB_LOG_COLORS", "true").lower() in ("true", "1", "yes")
    
    if show_time is None:
        show_time = os.getenv("MAGPYLIB_LOG_TIME", "true").lower() in ("true", "1", "yes")
    
    if sink is None:
        sink = sys.stderr
    
    # Build format string
    time_part = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " if show_time else ""
    format_str = (
        f"{time_part}"
        "<level>{level:^8}</level> | "
        "<cyan>{extra[module]}</cyan> | "
        "{level.icon:<2} {message}"
    )
    
    # Configure the logger
    logger.add(
        sink,
        level=level,
        format=format_str,
        colorize=enable_colors,
        filter=lambda record: (
            record["extra"].get("module", "").startswith("magpylib_material_response") or 
            record.get("name", "").startswith("magpylib_material_response")
        )
    )


def disable_logging() -> None:
    """Disable all logging output from the package."""
    logger.remove()
    logger.add(sink=lambda _: None, level="CRITICAL")  # Sink that does nothing


# Set up a default minimal configuration that doesn't output anything
# Users need to call configure_logging() to see logs
disable_logging()