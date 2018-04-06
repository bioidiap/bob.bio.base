'''Plots and measures for bob.bio.base'''

import click
import matplotlib.pyplot as mpl
import  bob.measure.script.figure as measure_figure
import bob.measure
from bob.measure import plot
from tabulate import tabulate

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

class Dic(measure_figure.PlotBase):
    ''' Handles the plotting of DIC

    Attributes
    ----------

    _semilogx: :obj:`bool`
        If true (default), X-axis will be semilog10
    _rank: :obj:`int`
        Rank to be used to plot DIC (default: 1)
    '''
    def __init__(self, ctx, scores, test, func_load):
        super(Dic, self).__init__(ctx, scores, test, func_load)
        self._semilogx = True if 'semilogx' not in ctx.meta else\
                ctx.meta['semilogx']
        self._rank = 1 if 'rank' not in ctx.meta else ctx.meta['rank']
        self._title = 'DIC'
        self._x_label = 'FAR'
        self._y_label = 'DIR'

    def compute(self, idx, dev_score, dev_file=None,
                test_score=None, test_file=None):
        ''' Plot DIC for dev and eval data using
        :py:func:`bob.measure.plot.detection_identification_curve`'''
        mpl.figure(1)
        if self._test:
            linestyle = '-' if not self._split else measure_figure.LINESTYLES[idx % 14]
            plot.detection_identification_curve(
                dev_score, rank=self._rank, logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('development', dev_file, idx)
            )
            linestyle = '--'
            if self._split:
                mpl.figure(2)
                linestyle = measure_figure.LINESTYLES[idx % 14]

            plot.detection_identification_curve(
                test_score, rank=self._rank, logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('test', test_file, idx)
            )
        else:
            rank = plot.detection_identification_curve(
                dev_score, rank=self._rank, logx=self._semilogx,
                color=self._colors[idx], linestyle=measure_figure.LINESTYLES[idx % 14],
                label=self._label('development', dev_file, idx)
            )

class Metrics(measure_figure.Metrics):
    ''' Compute metrics from score files'''
    def init_process(self):
        if self._criter == 'rr':
            self._thres = [None] * len(self.dev_names) if self._thres is None else \
                    self._thres

    def compute(self, idx, dev_score, dev_file=None,
                test_score=None, test_file=None):
        ''' Compute metrics for the given criteria'''
        headers = ['', 'Development %s' % dev_file]
        if self._test and test_score is not None:
            headers.append('Test % s' % test_file)
        if self._criter == 'rr':
            rr = bob.measure.recognition_rate(dev_score, self._thres[idx])
            dev_rr = "%.3f%%" % (100 * rr)
            raws = [['RR', dev_rr]]
            if self._test and test_score is not None:
                rr = bob.measure.recognition_rate(test_score, self._thres[idx])
                test_rr = "%.3f%%" % (100 * rr)
                raws[0].append(test_rr)
            click.echo(
                tabulate(raws, headers, self._tablefmt), file=self.log_file
            )
        elif self._criter == 'mindcf':
            if 'cost' in self._ctx.meta:
                cost = 0.99 if 'cost' not in self._ctx.meta else\
                        self._ctx.meta['cost']
            threshold = bob.measure.min_weighted_error_rate_threshold(
                dev_score[0], dev_score[1], cost
            ) if self._thres is None else self._thres[idx]
            if self._thres is None:
                click.echo(
                    "[minDCF - Cost:%f] Threshold on Development set `%s`: %e"\
                    % (cost, dev_file, threshold),
                    file=self.log_file
                )
            else:
                click.echo(
                    "[minDCF] User defined Threshold: %e" %  threshold,
                    file=self.log_file
                )
            # apply threshold to development set
            far, frr = bob.measure.farfrr(
                dev_score[0], dev_score[1], threshold
            )
            dev_far_str = "%.3f%%" % (100 * far)
            dev_frr_str = "%.3f%%" % (100 * frr)
            dev_mindcf_str = "%.3f%%" % ((cost * far + (1 - cost) * frr) * 100.)
            raws = [['FAR', dev_far_str],
                    ['FRR', dev_frr_str],
                    ['minDCF', dev_mindcf_str]]
            if self._test and test_score is not None:
                # apply threshold to development set
                far, frr = bob.measure.farfrr(
                    test_score[0], test_score[1], threshold
                )
                test_far_str = "%.3f%%" % (100 * far)
                test_frr_str = "%.3f%%" % (100 * frr)
                test_mindcf_str = "%.3f%%" % ((cost * far + (1 - cost) * frr) * 100.)
                raws[0].append(test_far_str)
                raws[1].append(test_frr_str)
                raws[2].append(test_mindcf_str)
            click.echo(
                tabulate(raws, headers, self._tablefmt), file=self.log_file
            )
        elif self._criter == 'cllr':
            cllr = bob.measure.calibration.cllr(dev_score[0], dev_score[1])
            min_cllr = bob.measure.calibration.min_cllr(
                dev_score[0], dev_score[1]
            )
            dev_cllr_str = "%.3f%%" % cllr
            dev_min_cllr_str = "%.3f%%" % min_cllr
            raws = [['Cllr', dev_cllr_str],
                    ['minCllr', dev_min_cllr_str]]
            if self._test and test_score is not None:
                cllr = bob.measure.calibration.cllr(test_score[0],
                                                    test_score[1])
                min_cllr = bob.measure.calibration.min_cllr(
                    test_score[0], test_score[1]
                )
                test_cllr_str = "%.3f%%" % cllr
                test_min_cllr_str = "%.3f%%" % min_cllr
                raws[0].append(test_cllr_str)
                raws[1].append(test_min_cllr_str)
                click.echo(
                    tabulate(raws, headers, self._tablefmt), file=self.log_file
                )
        else:
            super(Metrics, self).compute(
                idx, dev_score, dev_file, test_score, test_file
            )
