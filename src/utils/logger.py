"""Logging utility using loguru."""

import sys
from loguru import logger
from pathlib import Path

def setup_logger(log_dir: str = "outputs/logs"):
    """
    Configure loguru to log to both console and file.
    
    Args:
        log_dir: Directory to save log files
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    log_path = Path(log_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        str(log_path),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="1 week"
    )
    
    return logger
