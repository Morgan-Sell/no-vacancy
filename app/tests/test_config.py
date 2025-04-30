import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import Mock, patch

from config import FORMATTER, LOG_FILE, get_console_handler, get_file_handler


def test_get_console_handler():
    handler = get_console_handler()

    # Check if the handler is a StreamHandler
    assert isinstance(
        handler, logging.StreamHandler
    ), "Handler should be an instance of StreamHandler"

    # Check if the handler uses sys.stdout
    assert handler.stream == sys.stdout, "Handler stream should be sys.stdout"

    # Check if the handler has the correct formatter
    assert handler.formatter == FORMATTER, "Handler formatter is not set correctly"


def test_get_file_handler(tmpdir):
    # simulate scheduled log recording
    with patch("logging.handlers.TimedRotatingFileHandler") as mock_handler:
        mock_handler.return_value.baseFilename = LOG_FILE
        handler = get_file_handler()

        # Check if the handler is a TimedRotatingFileHandler
        assert isinstance(
            handler, TimedRotatingFileHandler
        ), "Handler should be a TimedRotatingFileHandler"

        # Check if the log level is set to WARNING
        assert (
            handler.level == logging.WARNING
        ), "Handler level should be set to WARNING"

        # Check if the formatter is correctly set
        assert handler.formatter == FORMATTER, "Handler formatter is not set correctly"

        # Check if the log file path is correct
        assert handler.baseFilename == LOG_FILE, "Log file path is incorrect"
