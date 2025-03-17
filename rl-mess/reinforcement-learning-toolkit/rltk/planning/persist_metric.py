import pickle
import sqlite3
from datetime import datetime
from multiprocessing import Queue
from typing import List

from .planning_metric import PlanningMetric

BULK_INSERT_SIZE = 5


def start(db: str, run_id: str, q: Queue):
    pm = PersistMetric(db, run_id, q)
    pm.to_disk()


start_msg = PlanningMetric(context_id='Writer', iter_num=0)
end_msg = PlanningMetric(context_id='Writer', iter_num=1)

def is_end_msg(msg):
    return msg.context_id == end_msg.context_id and msg.iter_num == end_msg.iter_num


class PersistMetric:
    def __init__(self, db, run_id, q):
        self._q = q
        self._conn = sqlite3.connect(db)
        self._run_id = run_id
        self._metrics_buf: List[PlanningMetric] = []

    def to_disk(self):
        while True:
            metric: PlanningMetric = self._q.get()
            self._save(metric)
            if is_end_msg(metric):
                break

    def _save(self, metric: PlanningMetric):
        sql = '''
        INSERT INTO planning_metrics
        VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        self._conn.execute(sql, [
            self._run_id,
            metric.context_id,
            datetime.now().isoformat(),
            metric.iter_num,
            pickle.dumps(metric.svals, pickle.HIGHEST_PROTOCOL),
            pickle.dumps(metric.qvals, pickle.HIGHEST_PROTOCOL),
            pickle.dumps(metric.pidist, pickle.HIGHEST_PROTOCOL)
        ])
        self._conn.commit()
