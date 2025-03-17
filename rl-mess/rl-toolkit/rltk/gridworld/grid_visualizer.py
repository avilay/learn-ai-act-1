import math

import plotter as pltr

from rltk import RlMetrics
from .grid_mdp import GridMDP


def visualize_svals(kit_name: str, mdp: GridMDP):
    rl_metrics = RlMetrics.instance()
    metric = rl_metrics.sval_metric
    lo, hi = metric.range(kit_name=kit_name)
    y_lo = float(math.floor(lo))
    y_hi = float(math.ceil(hi))

    frame = pltr.Frame(height_px=1500, width_px=1500)
    frame.layout(nrows=mdp.num_rows, ncols=mdp.num_cols)
    for state in mdp.states:
        # TODO: For terminal states plot a Text plot with red X
        yvals = []
        logs = metric.logs(kit_name=kit_name, state=str(state))
        for log in logs:
            yvals.append(log['value'])
        xvals = list(range(len(yvals)))
        chart = frame.create_chart()
        chart.title = str(state)
        chart.y_axis.limits = (y_lo, y_hi)
        line = pltr.Line(categories=xvals, values=yvals)
        chart.add(line)
    frame.show()
