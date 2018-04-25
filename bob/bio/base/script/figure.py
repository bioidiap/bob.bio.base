'''Plots and measures for bob.bio.base'''

import click
import matplotlib.pyplot as mpl
import  bob.measure.script.figure as measure_figure
import bob.measure
from bob.measure import plot
from tabulate import tabulate

class Roc(measure_figure.Roc):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = 'False Match Rate' if 'x_label' not in ctx.meta  or \
        ctx.meta['x_label'] is None else ctx.meta['x_label']
        self._y_label = '1 - False Non Match Rate' if 'y_label' not in \
        ctx.meta or ctx.meta['y_label'] is None else ctx.meta['y_label']

class Det(measure_figure.Det):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Det, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = 'False Match Rate' if 'x_label' not in ctx.meta or \
        ctx.meta['x_label'] is None else ctx.meta['x_label']
        self._y_label = 'False Non Match Rate' if 'y_label' not in ctx.meta or\
        ctx.meta['y_label'] is None else ctx.meta['y_label']

class Cmc(measure_figure.PlotBase):
    ''' Handles the plotting of Cmc '''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Cmc, self).__init__(ctx, scores, evaluation, func_load)
        self._semilogx = True if 'semilogx' not in ctx.meta else\
        ctx.meta['semilogx']
        self._title = self._title or 'CMC'
        self._x_label = self._x_label or 'Rank'
        self._y_label = self._y_label or 'Identification rate'
        self._max_R = 0

    def compute(self, idx, input_scores, input_names):
        ''' Plot CMC for dev and eval data using
        :py:func:`bob.measure.plot.cmc`'''
        mpl.figure(1)
        if self._eval:
            linestyle = '-' if not self._split else measure_figure.LINESTYLES[idx % 14]
            rank = plot.cmc(
                input_scores[0], logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('development', input_names[0], idx)
            )
            self._max_R = max(rank, self._max_R)
            linestyle = '--'
            if self._split:
                mpl.figure(2)
                linestyle = measure_figure.LINESTYLES[idx % 14]

            rank = plot.cmc(
                input_scores[1], logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('eval', input_names[1], idx)
            )
            self._max_R = max(rank, self._max_R)
        else:
            rank = plot.cmc(
                input_scores[0], logx=self._semilogx,
                color=self._colors[idx], linestyle=measure_figure.LINESTYLES[idx % 14],
                label=self._label('development', input_names[0], idx)
            )
            self._max_R = max(rank, self._max_R)

class Dir(measure_figure.PlotBase):
    ''' Handles the plotting of DIR curve'''
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Dir, self).__init__(ctx, scores, evaluation, func_load)
        self._semilogx = True if 'semilogx' not in ctx.meta else\
                ctx.meta['semilogx']
        self._rank = 1 if 'rank' not in ctx.meta else ctx.meta['rank']
        self._title = self._title or 'DIR curve'
        self._x_label = self._title or 'FAR'
        self._y_label = self._title or 'DIR'

    def compute(self, idx, input_scores, input_names):
        ''' Plot DIR for dev and eval data using
        :py:func:`bob.measure.plot.detection_identification_curve`'''
        mpl.figure(1)
        if self._eval:
            linestyle = '-' if not self._split else measure_figure.LINESTYLES[idx % 14]
            plot.detection_identification_curve(
                input_scores[0], rank=self._rank, logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('development', input_names[0], idx)
            )
            linestyle = '--'
            if self._split:
                mpl.figure(2)
                linestyle = measure_figure.LINESTYLES[idx % 14]

            plot.detection_identification_curve(
                input_scores[1], rank=self._rank, logx=self._semilogx,
                color=self._colors[idx], linestyle=linestyle,
                label=self._label('eval', input_names[1], idx)
            )
        else:
            plot.detection_identification_curve(
                input_scores[0], rank=self._rank, logx=self._semilogx,
                color=self._colors[idx], linestyle=measure_figure.LINESTYLES[idx % 14],
                label=self._label('development', input_names[0], idx)
            )

class Metrics(measure_figure.Metrics):
    ''' Compute metrics from score files'''
    def init_process(self):
        if self._criter == 'rr':
            self._thres = [None] * self.n_systems if self._thres is None else \
                    self._thres

    def compute(self, idx, input_scores, input_names):
        ''' Compute metrics for the given criteria'''
        title = self._titles[idx] if self._titles is not None else None
        headers = ['' or title, 'Development %s' % input_names[0]]
        if self._eval and input_scores[1] is not None:
            headers.append('eval % s' % input_names[1])
        if self._criter == 'rr':
            rr = bob.measure.recognition_rate(input_scores[0], self._thres[idx])
            dev_rr = "%.1f%%" % (100 * rr)
            raws = [['RR', dev_rr]]
            if self._eval and input_scores[1] is not None:
                rr = bob.measure.recognition_rate(input_scores[1], self._thres[idx])
                eval_rr = "%.1f%%" % (100 * rr)
                raws[0].append(eval_rr)
            click.echo(
                tabulate(raws, headers, self._tablefmt), file=self.log_file
            )
        elif self._criter == 'mindcf':
            if 'cost' in self._ctx.meta:
                cost = 0.99 if 'cost' not in self._ctx.meta else\
                        self._ctx.meta['cost']
            threshold = bob.measure.min_weighted_error_rate_threshold(
                input_scores[0][0], input_scores[0][1], cost
            ) if self._thres is None else self._thres[idx]
            if self._thres is None:
                click.echo(
                    "[minDCF - Cost:%f] Threshold on Development set `%s`: %e"\
                    % (cost, input_names[0], threshold),
                    file=self.log_file
                )
            else:
                click.echo(
                    "[minDCF] User defined Threshold: %e" %  threshold,
                    file=self.log_file
                )
            # apply threshold to development set
            far, frr = bob.measure.farfrr(
                input_scores[0][0], input_scores[0][1], threshold
            )
            dev_far_str = "%.1f%%" % (100 * far)
            dev_frr_str = "%.1f%%" % (100 * frr)
            dev_mindcf_str = "%.1f%%" % ((cost * far + (1 - cost) * frr) * 100.)
            raws = [['FAR', dev_far_str],
                    ['FRR', dev_frr_str],
                    ['minDCF', dev_mindcf_str]]
            if self._eval and input_scores[1] is not None:
                # apply threshold to development set
                far, frr = bob.measure.farfrr(
                    input_scores[1][0], input_scores[1][1], threshold
                )
                eval_far_str = "%.1f%%" % (100 * far)
                eval_frr_str = "%.1f%%" % (100 * frr)
                eval_mindcf_str = "%.1f%%" % ((cost * far + (1 - cost) * frr) * 100.)
                raws[0].append(eval_far_str)
                raws[1].append(eval_frr_str)
                raws[2].append(eval_mindcf_str)
            click.echo(
                tabulate(raws, headers, self._tablefmt), file=self.log_file
            )
        elif self._criter == 'cllr':
            cllr = bob.measure.calibration.cllr(input_scores[0][0],
                                                input_scores[0][1])
            min_cllr = bob.measure.calibration.min_cllr(
                input_scores[0][0], input_scores[0][1]
            )
            dev_cllr_str = "%.1f%%" % cllr
            dev_min_cllr_str = "%.1f%%" % min_cllr
            raws = [['Cllr', dev_cllr_str],
                    ['minCllr', dev_min_cllr_str]]
            if self._eval and input_scores[1] is not None:
                cllr = bob.measure.calibration.cllr(input_scores[1][0],
                                                    input_scores[1][1])
                min_cllr = bob.measure.calibration.min_cllr(
                    input_scores[1][0], input_scores[1][1]
                )
                eval_cllr_str = "%.1f%%" % cllr
                eval_min_cllr_str = "%.1f%%" % min_cllr
                raws[0].append(eval_cllr_str)
                raws[1].append(eval_min_cllr_str)
                click.echo(
                    tabulate(raws, headers, self._tablefmt), file=self.log_file
                )
        else:
            super(Metrics, self).compute(idx, input_scores, input_names)

class Hist(measure_figure.Hist):
    ''' Histograms for biometric scores '''

    def _setup_hist(self, neg, pos):
        self._title_base = 'Bio scores'
        self._density_hist(
            pos[0], label='Genuines', alpha=0.9, color='C2'
        )
        self._density_hist(
            neg[0], label='Zero-effort impostors', alpha=0.8, color='C0'
        )
