import os.path as path
import sqlite3
import sys
from multiprocessing import Process, Queue
from haikunator import Haikunator
from rltk.metrics.writer import Writer

from .persist_metric import start, start_msg, end_msg
from .planning_metric import PlanningMetric


def create(db: str):
    sql = '''
    CREATE TABLE planning_metrics
    (run_id text, context_id text, timestamp text, iter_num int, svals blob, qvals blob, pidist blob)
    '''
    if not path.exists(path.dirname(db)):
        print(f'Path {path.dirname(db)} to db does not exist!')
        sys.exit(1)

    conn = sqlite3.connect(db)
    conn.execute(sql)
    conn.commit()
    conn.close()


class SqlitePlanningMetricWriter(Writer):
    def __init__(self, db: str):
        if db != ':memory:' and not path.exists(db):
            create(db)
        self._q: Queue = Queue()
        self._q.put(start_msg)

        self._run_id: str = Haikunator().haikunate()

        self._p = Process(target=start, args=(db, self._run_id, self._q,))
        self._p.start()

    @property
    def run_id(self) -> str:
        return self._run_id

    def write(self, metric: PlanningMetric):
        self._q.put(metric)

    def close(self):
        self._q.put(end_msg)
