import logging
import sys
from typing import Optional


def setup_global_logging(verbose: bool = False) -> None:
    """Configure global logging for the entire application.

    This function sets up a root logger that will be used across all modules.
    The logging level is determined by the verbose flag:
    - If verbose is True: DEBUG level
    - If verbose is False: INFO level

    Parameters
    ----------
    verbose : bool, optional
        Whether to enable debug logging, by default False
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Remove any existing handlers to avoid duplicate logging
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the logging level based on verbose flag
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(console_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for a specific module.

    This function returns a logger that inherits from the root logger,
    ensuring consistent logging configuration across all modules.

    Parameters
    ----------
    name : Optional[str], optional
        The name of the module requesting the logger, by default None
        If None, returns the root logger

    Returns
    -------
    logging.Logger
        A logger instance configured with the global settings
    """
    return logging.getLogger(name)
