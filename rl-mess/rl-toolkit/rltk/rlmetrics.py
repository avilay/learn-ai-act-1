from rltk import Kit
from .core import StateValues


class RlMetrics:
    _singleton = None

    def __init__(self):
        self._kit = Kit.instance()
        self._sval_metric = None
        self._qval_metric = None
        self._pi_metric = None

    @classmethod
    def instance(cls):
        if not cls._singleton:
            cls._singleton = cls()
        return cls._singleton

    @property
    def sval_metric(self):
        if not self._sval_metric:
            self._sval_metric = self._kit.new_metric('svals', {'kit_name', 'state'})
        return self._sval_metric

    def log_svals(self, svals: StateValues):
        self.sval_metric.start_snapshot()
        for s, v in svals:
            self.sval_metric.log(kit_name=self._kit.name, state=str(s), value=v)
        self.sval_metric.stop_snapshot()

    def close(self):
        if self._sval_metric:
            self._sval_metric.close()
