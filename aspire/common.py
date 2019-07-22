import sys
import logging


# Define message formatter
class AspireLogFormatter(logging.Formatter):

    dbg_fmt = "[%(asctime)-15s][%(filename)s:%(lineno)03d]%(message)s"
    info_fmt = "[%(asctime)-15s] %(message)s"

    def __init__(self):
        super().__init__(fmt="%(levelno)d: %(msg)s", datefmt="%Y-%m-%d %H:%M:%S", style='%')

    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._style._fmt = AspireLogFormatter.dbg_fmt

        elif record.levelno == logging.INFO:
            self._style._fmt = AspireLogFormatter.info_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._style._fmt = format_orig

        return result


def configure_logger(logger, logfile, verbosity):
    """
    Set verbosity and output stream for the logger.
    args are command line arguments that contain the verbosity level and the
    output (stdout/file) of the logger.
    Configure the logger according to these parameters.
    """

    logger.handlers = []    # Remove current logger handlers.

    if verbosity == 0:
        logger.propagte = False
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    elif verbosity == 2:
        logger.setLevel(logging.DEBUG)

    logging_format = AspireLogFormatter()
    if logfile is not None:
        lh = logging.FileHandler(logfile)
        lh.setFormatter(logging_format)
        logger.addHandler(lh)

    lh = logging.StreamHandler(sys.stdout)
    lh.setFormatter(logging_format)
    logger.addHandler(lh)


# Define default logger. Can be overridden by configure_logger.
# Use stdout for output, at INFO level, with aspire message format
default_logger = logging.getLogger(__name__)
stdout_handler = logging.StreamHandler()

fmt = AspireLogFormatter()
stdout_handler.setFormatter(fmt)
default_logger.addHandler(stdout_handler)
default_logger.setLevel(logging.INFO)
