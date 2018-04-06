'''Plots and measures for bob.bio.base'''

import matplotlib.pyplot as mpl
import  bob.measure.script.figure as measure_figure
from bob.measure import plot

class Cmc(measure_figure.PlotBase):
    ''' Handles the plotting of Cmc

    Attributes
    ----------

    _semilogx: :obj:`bool`
        If true (default), X-axis will be semilog10
    '''
    def __init__(self, ctx, scores, test, func_load):
        super(Cmc, self).__init__(ctx, scores, test, func_load)
        self._semilogx = True if 'semilogx' not in ctx.meta else\
        ctx.meta['semilogx']
        self._title = 'CMC'
        self._x_label = 'Rank'
        self._y_label = 'Probability'
        self._max_R = 0

    def compute(self, idx, dev_score, dev_file=None,
                test_score=None, test_file=None):
        ''' Plot CMC for dev and eval data using
        :py:func:`bob.measure.plot.cmc`'''
        mpl.figure(1)
        if self._test:
            linestyle = '-' if not self._split else measure_figure.LINESTYLES[idx % 14]
            rank = plot.cmc(
                dev_score, logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('development', dev_file, idx)
            )
            self._max_R = max(rank, self._max_R)
            linestyle = '--'
            if self._split:
                mpl.figure(2)
                linestyle = measure_figure.LINESTYLES[idx % 14]

            rank = plot.cmc(
                test_score, logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('test', test_file, idx)
            )
            self._max_R = max(rank, self._max_R)
        else:
            rank = plot.cmc(
                dev_score, logx=self._semilogx,
                color=self._colors[idx], linestyle=measure_figure.LINESTYLES[idx % 14],
                label=self._label('development', dev_file, idx)
            )
            self._max_R = max(rank, self._max_R)

    def end_process(self):
        ''' Set custom default if not axis limits provided '''
        if self._axlim is None:
            self._axlim = [0, self._max_R, -0.01, 1.01]
        super(Cmc, self).end_process()
