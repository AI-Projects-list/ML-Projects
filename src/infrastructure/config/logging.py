"""Logging configuration and setup."""

import sys
from pathlib import Path

from loguru import logger

from src.infrastructure.config.settings import Settings


def setup_logging(settings: Settings) -> None:
    """
    Configure logging for the application.
    
    Args:
        settings: Application settings
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=settings.log.format,
        level=settings.log.level,
        colorize=True,
    )
    
    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "ml_pipeline_{time:YYYY-MM-DD}.log",
        format=settings.log.format,
        level=settings.log.level,
        rotation=settings.log.rotation,
        retention=settings.log.retention,
        compression="zip",
    )
    
    logger.info("Logging configured successfully")
