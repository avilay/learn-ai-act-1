from typing import Any
from .writer import Writer, NullWriter

class Logger:
    def __init__(self) -> None:
        self._writer: Writer = NullWriter()

    def set_writer(self, writer: Writer) -> None:
        self._writer = writer

    def log(self, metric: Any) -> None:
        self._writer.write(metric)


_logger = Logger()

def get_logger():
    return _logger

def log(metric):
    _logger.log(metric)

def set_writer(writer: Writer):
    _logger.set_writer(writer)
