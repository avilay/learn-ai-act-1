from .writer import Writer, NullWriter
from .serializable import Serializable


class Logger:
    def __init__(self) -> None:
        self._writer: Writer = NullWriter()

    def set_writer(self, writer: Writer) -> None:
        self._writer = writer

    def log(self, metric: Serializable) -> None:
        self._writer.write(metric)


_logger = Logger()


def get_logger():
    return _logger


def log(metric: Serializable):
    _logger.log(metric)


def set_writer(writer: Writer):
    _logger.set_writer(writer)
