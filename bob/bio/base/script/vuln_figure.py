'''Runs error analysis on score sets, outputs metrics and plots'''

import click
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as mpl
import bob.measure.script.figure as measure_figure
from bob.measure.utils import get_thres, remove_nan
from bob.measure import (
    frr_threshold, far_threshold, farfrr, f_score, roc_auc_score,
    ppndf, min_weighted_error_rate_threshold, precision_recall,
)
from bob.measure import plot
from . import error_utils
import logging

logger = logging.getLogger(__name__)


def clean_scores(input_scores):
    """Returns a dict with each scores groups cleaned

    Parameters
    ----------
    input_scores: dict

    Returns
    -------
    clean_scores: dict
    """
    clean_scores = {}
    for key, scores in input_scores.items():
        clean_scores[key], _, _ = remove_nan(scores)
    return clean_scores

def far_for_threshold(scores, threshold):
    """Returns the ratio of scores over the threshold value

    Corresponds to FAR for negative scores, and 1-FRR for positive scores.
    """
    return (scores >= threshold).sum() / len(scores)

def frr_for_threshold(scores, threshold):
    """Returns the ratio of scores over the threshold value

    Corresponds to FRR for positive scores, and 1-FAR for negative scores.
    """
    return (scores < threshold).sum() / len(scores)


class Metrics(measure_figure.Metrics):
    """ Compute metrics from score files

    Attributes
    ----------

    names: dict {str:str}
        pairs of metrics keys and corresponding row titles to display.
    """

    def __init__(
        self,
        ctx,
        scores,
        evaluation,
        func_load,
        names={
            "fta": "Licit Failure to Acquire",
            "fmr": "Licit False Match Rate",
            "fnmr": "Licit False Non Match Rate",
            "far": "Licit False Accept Rate",
            "frr": "Licit False Reject Rate",
            "hter": "Licit Half Total Error Rate",
            "iapmr": "Attack Presentation Match Rate",
        },
        **kwargs,
    ):
        super(Metrics, self).__init__(ctx, scores, evaluation, func_load, names, **kwargs)

    def _get_all_metrics(self, idx, input_scores, input_names):
        """ Compute all metrics for dev and eval scores"""
        # Parse input and remove/count failed samples (NaN)
        dev_neg, dev_neg_na, dev_neg_count = remove_nan(input_scores[0]['licit_neg'])
        dev_pos, dev_pos_na, dev_pos_count = remove_nan(input_scores[0]['licit_pos'])
        dev_spoof, dev_spoof_na, dev_spoof_count = remove_nan(input_scores[0]['spoof'])
        dev_fta = (dev_neg_na + dev_pos_na + dev_spoof_na) / (dev_neg_count + dev_pos_count + dev_spoof_count)
        if self._eval:
            eval_neg, eval_neg_na, eval_neg_count = remove_nan(input_scores[1]['licit_neg'])
            eval_pos, eval_pos_na, eval_pos_count = remove_nan(input_scores[1]['licit_pos'])
            eval_spoof, eval_spoof_na, eval_spoof_count = remove_nan(input_scores[1]['spoof'])
            eval_fta = (eval_neg_na + eval_pos_na + eval_spoof_na) / (eval_neg_count + eval_pos_count + eval_spoof_count)
        dev_file = input_names[0]

        # Compute threshold on dev set
        threshold = (
            self.get_thres(self._criterion, dev_neg, dev_pos, self._far)
            if self._thres is None
            else self._thres[idx]
        )

        title = self._legends[idx] if self._legends is not None else None
        if self._thres is None:
            far_str = ""
            if self._criterion == "far" and self._far is not None:
                far_str = str(self._far)
            click.echo(
                f"[Min. criterion: {self._criterion.upper()} {far_str}] "
                f"Threshold on Development set `{title or dev_file}`: {threshold:e}",
                file=self.log_file,
            )
        else:
            click.echo(
                "[Min. criterion: user provided] Threshold on "
                f"Development set `{dev_file or title}`: {threshold:e}",
                file=self.log_file,
            )

        res = []
        res.append(self._strings(self._numbers(dev_neg, dev_pos, dev_spoof, threshold, dev_fta)))

        if self._eval:
            # computes statistics for the eval set based on the threshold a
            # priori computed on the dev set
            res.append(
                self._strings(self._numbers(eval_neg, eval_pos, eval_spoof, threshold, eval_fta))
            )
        else:
            res.append(None)

        return res

    def _numbers(self, neg, pos, spoof, threshold, fta):
        """Computes each metric value
        """
        # fpr and fnr
        fmr, fnmr = farfrr(neg, pos, threshold)
        hter = (fmr + fnmr) / 2.0
        far = fmr * (1 - fta)
        frr = fta + fnmr * (1 - fta)

        ni = neg.shape[0]  # number of impostors
        fm = int(round(fmr * ni))  # number of false accepts
        nc = pos.shape[0]  # number of clients
        fnm = int(round(fnmr * nc))  # number of false rejects

        # precision and recall
        precision, recall = precision_recall(neg, pos, threshold)

        # f_score
        f1_score = f_score(neg, pos, threshold, 1)

        # AUC ROC
        auc = roc_auc_score(neg, pos)
        auc_log = roc_auc_score(neg, pos, log_scale=True)

        # IAPMR at threshold
        iapmr = far_for_threshold(spoof, threshold)
        spoof_total = len(spoof)
        spoof_match = int(round(iapmr * spoof_total))

        return {
            "fta": fta,
            "fmr": fmr,
            "fnmr": fnmr,
            "hter": hter,
            "far": far,
            "frr": frr,
            "fm": fm,
            "ni": ni,
            "fnm": fnm,
            "nc": nc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "auc": auc,
            "auc_log": auc_log,
            "iapmr": iapmr,
            "spoof_match": spoof_match,
            "spoof_total": spoof_total,
        }

    def _strings(self, metrics):
        """Formats the metrics values into strings
        """
        return {
            "fta": f"{100 * metrics['fta']:.{self._decimal}f}%",
            "fmr": f"{100 * metrics['fmr']:.{self._decimal}f}% ({metrics['fm']}/{metrics['ni']})",
            "fnmr": f"{100 * metrics['fnmr']:.{self._decimal}f}% ({metrics['fnm']}/{metrics['nc']})",
            "far": f"{100 * metrics['far']:.{self._decimal}f}%",
            "frr": f"{100 * metrics['frr']:.{self._decimal}f}%",
            "hter": f"{100 * metrics['hter']:.{self._decimal}f}%",
            "precision": f"{metrics['precision']:.{self._decimal}f}",
            "recall": f"{metrics['recall']:.{self._decimal}f}",
            "f1_score": f"{metrics['f1_score']:.{self._decimal}f}",
            "auc": f"{metrics['auc']:.{self._decimal}f}",
            "auc_log": f"{metrics['auc_log']:.{self._decimal}f}",
            "iapmr": f"{100 * metrics['iapmr']:.{self._decimal}f}% ({metrics['spoof_match']}/{metrics['spoof_total']})",
        }

    def compute(self, idx, input_scores, input_names):
        """ Compute metrics thresholds and tables for given system inputs
        """
        # Title and headers
        title = self._legends[idx] if self._legends is not None else None
        headers = ["" or title, "Dev. %s" % input_names[0]]
        if self._eval and input_scores[1] is not None:
            headers.append("eval % s" % input_names[1])

        # Tables rows
        all_metrics = self._get_all_metrics(idx, input_scores, input_names)
        headers = [" " or title, "Development"]

        rows = []
        for key, name in self.names.items():
            if key not in all_metrics[0]:
                logger.warning(f"{key} not present in metrics.")
            rows.append([name, all_metrics[0].get(key,"N/A")])

        if self._eval:
            # computes statistics for the eval set based on the threshold a
            # priori
            headers.append("Evaluation")
            for row, key in zip(rows, self.names.keys()):
                row.append(all_metrics[1].get(key,"N/A"))

        click.echo(tabulate(rows, headers, self._tablefmt), file=self.log_file)


def _iapmr_dot(threshold, iapmr, real_data, **kwargs):
    # plot a dot on threshold versus IAPMR line and show IAPMR as a number
    axlim = mpl.axis()
    mpl.plot(threshold, 100. * iapmr, 'o', color='C3', **kwargs)
    if not real_data:
        mpl.annotate(
            'IAPMR at\noperating point',
            xy=(threshold, 100. * iapmr),
            xycoords='data',
            xytext=(0.85, 0.6),
            textcoords='axes fraction',
            color='black',
            size='large',
            arrowprops=dict(facecolor='black', shrink=0.05, width=2),
            horizontalalignment='center',
            verticalalignment='top',
        )
    else:
        mpl.text(threshold + (threshold - axlim[0]) / 12, 100. * iapmr,
                 '%.1f%%' % (100. * iapmr,), color='C3')


def _iapmr_line_plot(scores, n_points=100, **kwargs):
    axlim = mpl.axis()
    step = (axlim[1] - axlim[0]) / float(n_points)
    thres = [(k * step) + axlim[0] for k in range(2, n_points - 1)]
    mix_prob_y = []
    for k in thres:
        mix_prob_y.append(100. * error_utils.calc_pass_rate(k, scores))

    mpl.plot(thres, mix_prob_y, label='IAPMR', color='C3', **kwargs)


def _iapmr_plot(scores, threshold, iapmr, real_data, **kwargs):
    _iapmr_dot(threshold, iapmr, real_data, **kwargs)
    _iapmr_line_plot(scores, n_points=100, **kwargs)


class HistVuln(measure_figure.Hist):
    ''' Histograms for vulnerability '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(HistVuln, self).__init__(
            ctx, scores, evaluation, func_load, nhist_per_system=3)

    def _setup_hist(self, neg, pos):
        self._title_base = ' '
        self._density_hist(pos[0], n=0, label='Genuine', color='C2')
        self._density_hist(
            neg[0], n=1, label='Zero-effort impostors', alpha=0.8, color='C0'
        )
        self._density_hist(
            pos[1], n=2, label='Presentation attack', alpha=0.4, color='C7',
            hatch='\\\\'
        )

    def _get_neg_pos_thres(self, idx, input_scores, input_names):
        """Get scores and threshold for the given system at index idx for vuln

        Returns
        -------
        dev_neg, dev_pos, eval_neg, eval_pos: list of arrays
            The scores negatives and positives for each set. Each element
            contains two lists: licit [0] and spoof [1]
        threshold: int
            The value of the threshold computed on the `dev` set licit scores.
        """

        dev_scores = clean_scores(input_scores[0])
        if self._eval:
            eval_scores = clean_scores(input_scores[1])
        else:
            eval_scores = {'licit_neg':[], 'licit_pos':[], 'spoof':[]}

        threshold = (
            get_thres(self._criterion, dev_scores['licit_neg'], dev_scores['licit_pos'])
            if self._thres is None
            else self._thres[idx]
        )
        return (
            [dev_scores['licit_neg'], ],
            [dev_scores['licit_pos'], dev_scores['spoof']],
            [eval_scores['licit_neg'], ],
            [eval_scores['licit_pos'], eval_scores['spoof']],
            threshold
        )


    def _lines(self, threshold, label, neg, pos, idx, **kwargs):
        spoof = pos[1]
        neg = neg[0]
        pos = pos[0]
        # plot EER treshold vertical line
        super(HistVuln, self)._lines(threshold, label, neg, pos, idx, **kwargs)

        if 'iapmr_line' not in self._ctx.meta or self._ctx.meta['iapmr_line']:
            # Plot iapmr_line (accepted PA vs threshold)
            iapmr = far_for_threshold(spoof, threshold)
            ax2 = mpl.twinx()
            # we never want grid lines on axis 2
            ax2.grid(False)
            real_data = True if 'real_data' not in self._ctx.meta else \
                self._ctx.meta['real_data']
            _iapmr_plot(spoof, threshold, iapmr, real_data=real_data)
            n = idx % self._step_print
            col = n % self._ncols
            rest_print = self.n_systems - \
                int(idx / self._step_print) * self._step_print
            if col == self._ncols - 1 or n == rest_print - 1:
                ax2.set_ylabel("IAPMR (%)", color='C3')
            ax2.tick_params(axis='y', colors='C3')
            ax2.yaxis.label.set_color('C3')
            ax2.spines['right'].set_color('C3')


class VulnPlot(measure_figure.PlotBase):
    '''Base class for vulnerability analysis plots'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(VulnPlot, self).__init__(ctx, scores, evaluation, func_load)
        mpl.rcParams['figure.constrained_layout.use'] = self._clayout
        self._nlegends = ctx.meta.get('legends_ncol', 3)

    def end_process(self):
        '''Close pdf '''
        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
           ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    def _plot_legends(self):
        # legends for all axes
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
        if self._disp_legend:
            mpl.gca().legend(lines, labels, loc=self._legend_loc,
                             ncol=self._nlegends, fancybox=True,
                             framealpha=0.5)


class Epc(VulnPlot):
    ''' Handles the plotting of EPC '''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        self._iapmr = True if 'iapmr' not in self._ctx.meta else \
            self._ctx.meta['iapmr']
        self._title = self._title or ('EPC and IAPMR' if self._iapmr else
                                      'EPC')
        self._x_label = self._x_label or r"Weight $\beta$"
        self._y_label = self._y_label or "HTER (%)"
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1

        if self._min_arg != 4:
            raise click.BadParameter("You must provide 4 scores files:{licit,"
                                     "spoof}/{dev,eval}")

    def compute(self, idx, input_scores, input_names):
        ''' Plot EPC for vuln'''
        # extract pos and negative and remove NaNs
        neg_list, pos_list, _ = get_fta_list([s for l in input_scores for s in l])
        licit_dev_neg, licit_dev_pos = neg_list[0], pos_list[0]
        licit_eval_neg, licit_eval_pos = neg_list[1], pos_list[1]
        spoof_eval_neg = neg_list[3]

        mpl.gcf().clear()
        mpl.grid()
        logger.info(f"EPC using {input_names[0]} and {input_names[1]}")
        plot.epc(
            licit_dev_neg, licit_dev_pos, licit_eval_neg, licit_eval_pos,
            self._points,
            color='C0', linestyle=self._linestyles[idx],
            label=self._label('HTER (licit)', idx)
        )
        mpl.xlabel(self._x_label)
        mpl.ylabel(self._y_label)
        if self._iapmr:
            ax1 = mpl.gca()
            mpl.gca().set_axisbelow(True)
            prob_ax = mpl.gca().twinx()
            step = 1.0 / float(self._points)
            thres = [float(k * step) for k in range(self._points)]
            thres.append(1.0)
            apply_thres = [min_weighted_error_rate_threshold(
                licit_dev_neg, licit_dev_pos, t) for t in thres]
            mix_prob_y = []
            for k in apply_thres:
                mix_prob_y.append(
                    100. * error_utils.calc_pass_rate(k, spoof_eval_neg)
                )

            logger.info(f"IAPMR in EPC plot using {input_names[0]}, {input_names[1]}")
            mpl.plot(
                thres, mix_prob_y, label=self._label('IAPMR (spoof)', idx), color='C3'
            )

            prob_ax.set_yticklabels(prob_ax.get_yticks())
            prob_ax.tick_params(axis='y', colors='C3')
            prob_ax.yaxis.label.set_color('C3')
            prob_ax.spines['right'].set_color('C3')
            ylabels = prob_ax.get_yticks()
            prob_ax.yaxis.set_ticklabels(["%.0f" % val for val in ylabels])
            prob_ax.set_ylabel('IAPMR (%)', color='C3')
            prob_ax.set_axisbelow(True)
            ax1.yaxis.label.set_color('C0')
            ax1.tick_params(axis='y', colors='C0')
            ax1.spines['left'].set_color('C0')

        title = self._legends[idx] if self._legends is not None else self._title
        if title.replace(' ', ''):
            mpl.title(title)
        # legends for all axes
        self._plot_legends()
        mpl.xticks(rotation=self._x_rotation)
        self._pdf_page.savefig(mpl.gcf())


class Epsc(VulnPlot, measure_figure.GridSubplot):
    ''' Handles the plotting of EPSC '''

    def __init__(self, ctx, scores, evaluation, func_load,
                 criteria, var_param, fixed_params):
        super(Epsc, self).__init__(ctx, scores, evaluation, func_load)
        self._iapmr = False if 'iapmr' not in self._ctx.meta else \
            self._ctx.meta['iapmr']
        self._wer = True if 'wer' not in self._ctx.meta else \
            self._ctx.meta['wer']
        self._criteria = criteria or 'eer'
        self._var_param = var_param or "omega"
        self._fixed_params = fixed_params or [0.5]
        self._titles = ctx.meta.get('titles', []) * 2
        self._eval = True  # always eval data with EPSC
        self._split = False
        self._nb_figs = 1
        self._sampling = ctx.meta.get('sampling', 5)
        mpl.grid(True)
        self._axis1 = None
        self._axis2 = None

        if self._min_arg != 4:
            raise click.BadParameter("You must provide 4 scores files:{licit,"
                                     "spoof}/{dev,eval}")

        self._ncols = 1 if self._iapmr else 0
        self._ncols += 1 if self._wer else 0

    def compute(self, idx, input_scores, input_names):
        ''' Plot EPSC for vuln'''
        licit_dev_neg = input_scores[0][0]
        licit_dev_pos = input_scores[0][1]
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_dev_neg = input_scores[2][0]
        spoof_dev_pos = input_scores[2][1]
        spoof_eval_neg = input_scores[3][0]
        spoof_eval_pos = input_scores[3][1]
        merge_sys = (self._fixed_params is None or
                     len(self._fixed_params) == 1) and self.n_systems > 1
        legend = ''
        if self._legends is not None and idx < len(self._legends):
            legend = self._legends[idx]
        elif self.n_systems > 1:
            legend = 'Sys%d' % (idx + 1)

        if not merge_sys or idx == 0:
            # axes should only be created once
            self._axis1 = self.create_subplot(0)
            if self._ncols == 2:
                self._axis2 = self.create_subplot(1)
            else:
                self._axis2 = self._axis1
        points = 10
        for pi, fp in enumerate(self._fixed_params):
            if merge_sys:
                assert pi == 0
                pi = idx
            if self._var_param == 'omega':
                omega, beta, thrs = error_utils.epsc_thresholds(
                    licit_dev_neg,
                    licit_dev_pos,
                    spoof_dev_neg,
                    spoof_dev_pos,
                    points=points,
                    criteria=self._criteria,
                    beta=fp)
            else:
                omega, beta, thrs = error_utils.epsc_thresholds(
                    licit_dev_neg,
                    licit_dev_pos,
                    spoof_dev_neg,
                    spoof_dev_pos,
                    points=points,
                    criteria=self._criteria,
                    omega=fp
                )

            errors = error_utils.all_error_rates(
                licit_eval_neg, licit_eval_pos, spoof_eval_neg,
                spoof_eval_pos, thrs, omega, beta
            )  # error rates are returned in a list in the
            # following order: frr, far, IAPMR, far_w, wer_w

            mpl.sca(self._axis1)
            # between the negatives (impostors and Presentation attacks)
            base = r"(%s) " % legend if legend.strip() else ""
            if self._wer:
                set_title = self._titles[idx] if self._titles is not None and \
                    len(self._titles) > idx else None
                display = set_title.replace(' ', '') if set_title is not None\
                    else True
                wer_title = set_title or ""
                if display:
                    mpl.title(wer_title)
                if self._var_param == 'omega':
                    label = r"%s$\beta=%.1f$" % (base, fp)
                    mpl.plot(
                        omega, 100. * errors[4].flatten(),
                        color=self._colors[pi], linestyle='-', label=label)
                    mpl.xlabel(self._x_label or r"Weight $\omega$")
                else:
                    label = r"%s$\omega=%.1f$" % (base, fp)
                    mpl.plot(
                        beta, 100. * errors[4].flatten(),
                        color=self._colors[pi], linestyle='-', label=label)
                    mpl.xlabel(self._x_label or r"Weight $\beta$")
                mpl.ylabel(self._y_label or r"WER$_{\omega,\beta}$ (%)")

            if self._iapmr:
                mpl.sca(self._axis2)
                set_title = self._titles[idx + self.n_systems] \
                    if self._titles is not None and \
                    len(self._titles) > self.n_systems + idx else None
                display = set_title.replace(' ', '') if set_title is not None\
                    else True
                iapmr_title = set_title or ""
                if display:
                    mpl.title(iapmr_title)
                if self._var_param == 'omega':
                    label = r"$%s $\beta=%.1f$" % (base, fp)
                    mpl.plot(
                        omega, 100. * errors[2].flatten(),
                        color=self._colors[pi], linestyle='-', label=label
                    )
                    mpl.xlabel(self._x_label or r"Weight $\omega$")
                else:
                    label = r"%s $\omega=%.1f$" % (base, fp)
                    mpl.plot(
                        beta, 100. * errors[2].flatten(), linestyle='-',
                        color=self._colors[pi], label=label
                    )
                    mpl.xlabel(self._x_label or r"Weight $\beta$")

                mpl.ylabel(self._y_label or r"IAPMR  (%)")
                self._axis2.set_xticklabels(self._axis2.get_xticks())
                self._axis2.set_yticklabels(self._axis2.get_yticks())

        self._axis1.set_xticklabels(self._axis1.get_xticks())
        self._axis1.set_yticklabels(self._axis1.get_yticks())
        mpl.xticks(rotation=self._x_rotation)
        if self._fixed_params is None or len(self._fixed_params) > 1 or \
           idx == self.n_systems - 1:
            self.finalize_one_page()


class Epsc3D(Epsc):
    ''' 3D EPSC plots for vuln'''

    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_dev_neg = input_scores[0][0]
        licit_dev_pos = input_scores[0][1]
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_dev_neg = input_scores[2][0]
        spoof_dev_pos = input_scores[2][1]
        spoof_eval_neg = input_scores[3][0]
        spoof_eval_pos = input_scores[3][1]

        title = self._legends[idx] if self._legends is not None else "3D EPSC"

        mpl.rcParams.pop('key', None)

        mpl.gcf().clear()
        mpl.gcf().set_constrained_layout(self._clayout)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        points = self._sampling or 5

        omega, beta, thrs = error_utils.epsc_thresholds(
            licit_dev_neg,
            licit_dev_pos,
            spoof_dev_neg,
            spoof_dev_pos,
            points=points,
            criteria=self._criteria)

        errors = error_utils.all_error_rates(
            licit_eval_neg, licit_eval_pos, spoof_eval_neg, spoof_eval_pos,
            thrs, omega, beta
        )
        # error rates are returned in a list as 2D numpy.ndarrays in
        # the following order: frr, far, IAPMR, far_w, wer_wb, hter_wb
        wer_errors = 100 * errors[2 if self._iapmr else 4]

        ax1 = mpl.gcf().add_subplot(111, projection='3d')

        W, B = np.meshgrid(omega, beta)

        ax1.plot_wireframe(
            W, B, wer_errors, cmap=cm.coolwarm, antialiased=False
        )  # surface

        if self._iapmr:
            ax1.azim = -30
            ax1.elev = 50

        ax1.set_xlabel(self._x_label or r"Weight $\omega$")
        ax1.set_ylabel(self._y_label or r"Weight $\beta$")
        ax1.set_zlabel(
            r"WER$_{\omega,\beta}$ (%)" if self._wer else "IAPMR (%)"
        )

        if title.replace(' ', ''):
            mpl.title(title)

        ax1.set_xticklabels(ax1.get_xticks())
        ax1.set_yticklabels(ax1.get_yticks())
        ax1.set_zticklabels(ax1.get_zticks())

        self._pdf_page.savefig()


class BaseVulnDetRoc(VulnPlot):
    '''Base for DET and ROC'''

    def __init__(self, ctx, scores, evaluation, func_load, real_data,
                 no_spoof):
        super(BaseVulnDetRoc, self).__init__(
            ctx, scores, evaluation, func_load)
        self._no_spoof = no_spoof
        self._fnmrs_at = ctx.meta.get('fnmr', [])
        self._real_data = True if real_data is None else real_data
        self._legend_loc = None
        self._min_dig = -4 if self._min_dig is None else self._min_dig

    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_neg = input_scores[0][0]
        licit_pos = input_scores[0][1]
        spoof_neg = input_scores[1][0]
        spoof_pos = input_scores[1][1]
        LOGGER.info("FNMR licit using %s", input_names[0])
        self._plot(
            licit_neg,
            licit_pos,
            self._points,
            color='C0',
            linestyle='-',
            label=self._label("Licit scenario", idx)
        )
        if not self._no_spoof and spoof_neg is not None:
            ax1 = mpl.gca()
            ax2 = ax1.twiny()
            ax2.set_xlabel('IAPMR (%)', color='C3')
            ax2.set_xticklabels(ax2.get_xticks())
            ax2.tick_params(axis='x', colors='C3')
            ax2.xaxis.label.set_color('C3')
            ax2.spines['top'].set_color('C3')
            ax2.spines['bottom'].set_color('C0')
            ax1.xaxis.label.set_color('C0')
            ax1.tick_params(axis='x', colors='C0')
            LOGGER.info("Spoof IAPMR using %s", input_names[1])
            self._plot(
                spoof_neg,
                spoof_pos,
                self._points,
                color='C3',
                linestyle=':',
                label=self._label("Spoof scenario", idx)
            )
            mpl.sca(ax1)

        if self._fnmrs_at is None:
            return

        for line in self._fnmrs_at:
            thres_baseline = frr_threshold(licit_neg, licit_pos, line)

            axlim = mpl.axis()

            farfrr_licit, farfrr_licit_det = self._get_farfrr(
                licit_neg, licit_pos,
                thres_baseline
            )
            if farfrr_licit is None:
                return

            farfrr_spoof, farfrr_spoof_det = self._get_farfrr(
                spoof_neg, spoof_pos,
                frr_threshold(spoof_neg, spoof_pos, farfrr_licit[1])
            )

            if not self._real_data:
                mpl.axhline(
                    y=farfrr_licit_det[1],
                    xmin=axlim[2],
                    xmax=axlim[3],
                    color='k',
                    linestyle='--',
                    label="%s @ EER" % self._y_label)
            else:
                mpl.axhline(
                    y=farfrr_licit_det[1],
                    xmin=axlim[0],
                    xmax=axlim[1],
                    color='k',
                    linestyle='--',
                    label="%s = %.2f%%" %
                    ('FMNR', farfrr_licit[1] * 100))

            if not self._real_data:
                label_licit = '%s @ operating point' % self._x_label
                label_spoof = 'IAPMR @ operating point'
            else:
                label_licit = 'FMR=%.2f%%' % (farfrr_licit[0] * 100)
                label_spoof = 'IAPMR=%.2f%%' % (farfrr_spoof[0] * 100)

            mpl.plot(
                farfrr_licit_det[0],
                farfrr_licit_det[1],
                'o',
                color='C0',
                label=label_licit
            )  # FAR point, licit scenario
            mpl.plot(
                farfrr_spoof_det[0],
                farfrr_spoof_det[1],
                'o',
                color='C3',
                label=label_spoof
            )  # FAR point, spoof scenario

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures and
        close pdf is needed '''
        # only for plots

        if self._title.replace(' ', ''):
            mpl.title(self._title, y=1.15)
        mpl.xlabel(self._x_label)
        mpl.ylabel(self._y_label)
        mpl.grid(True, color=self._grid_color)
        lines = []
        labels = []
        for ax in mpl.gcf().get_axes():
            li, la = ax.get_legend_handles_labels()
            lines += li
            labels += la
            mpl.sca(ax)
            self._set_axis()
            fig = mpl.gcf()
            mpl.xticks(rotation=self._x_rotation)
            mpl.tick_params(axis='both', which='major', labelsize=6)
        if self._disp_legend:
            mpl.gca().legend(
                lines, labels, loc=self._legend_loc, fancybox=True,
                framealpha=0.5
            )
        self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
                ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()

    def _get_farfrr(self, x, y, thres):
        return None, None

    def _plot(self, x, y, points, **kwargs):
        pass


class DetVuln(BaseVulnDetRoc):
    '''DET for vuln'''

    def __init__(self, ctx, scores, evaluation, func_load, real_data,
                 no_spoof):
        super(DetVuln, self).__init__(ctx, scores, evaluation, func_load,
                                      real_data, no_spoof)
        self._x_label = self._x_label or "FMR (%)"
        self._y_label = self._y_label or "FNMR (%)"
        add = ''
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        self._title = self._title or ('DET: LICIT' + add)
        self._legend_loc = self._legend_loc or 'upper right'

    def _set_axis(self):
        if self._axlim is not None and None not in self._axlim:
            plot.det_axis(self._axlim)
        else:
            plot.det_axis([0.01, 99, 0.01, 99])

    def _get_farfrr(self, x, y, thres):
        points = farfrr(x, y, thres)
        return points, [ppndf(i) for i in points]

    def _plot(self, x, y, points, **kwargs):
        LOGGER.info("Plot DET")
        plot.det(
            x, y, points,
            color=kwargs.get('color'),
            linestyle=kwargs.get('linestyle'),
            label=kwargs.get('label')
        )


class RocVuln(BaseVulnDetRoc):
    '''ROC for vuln'''

    def __init__(self, ctx, scores, evaluation, func_load, real_data, no_spoof):
        super(RocVuln, self).__init__(ctx, scores, evaluation, func_load,
                                      real_data, no_spoof)
        self._x_label = self._x_label or "FMR"
        self._y_label = self._y_label or "1 - FNMR"
        self._semilogx = ctx.meta.get('semilogx', True)
        add = ''
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        self._title = self._title or ('ROC: LICIT' + add)
        best_legend = 'lower right' if self._semilogx else 'upper right'
        self._legend_loc = self._legend_loc or best_legend

        logger.info("Plotting ROC")
        plot.roc(
            x, y,
            npoints=self._points,
            CAR=self._semilogx,
            min_far=self._min_dig,
            color=kwargs.get('color'),
            linestyle=kwargs.get('linestyle'),
            label=kwargs.get('label'),
        )

    def _get_farfrr(self, x, y, thres):
        points = farfrr(x, y, thres)
        points2 = (points[0], 1 - points[1])
        return points, points2


class FmrIapmr(VulnPlot):
    '''FMR vs IAPMR'''

    def __init__(self, ctx, scores, evaluation, func_load):
        super(FmrIapmr, self).__init__(ctx, scores, evaluation, func_load)
        self._eval = True  # Always ask for eval data
        self._split = False
        self._nb_figs = 1
        self._semilogx = ctx.meta.get('semilogx', False)
        if self._min_arg != 4:
            raise click.BadParameter("You must provide 4 scores files:{licit,"
                                     "spoof}/{dev,eval}")

    def compute(self, idx, input_scores, input_names):
        ''' Implements plots'''
        licit_eval_neg = input_scores[1][0]
        licit_eval_pos = input_scores[1][1]
        spoof_eval_neg = input_scores[3][0]
        fmr_list = np.linspace(0, 1, 100)
        iapmr_list = []
        for i, fmr in enumerate(fmr_list):
            thr = far_threshold(licit_eval_neg, licit_eval_pos, fmr)
            iapmr_list.append(farfrr(spoof_eval_neg, licit_eval_pos, thr)[0])
            # re-calculate fmr since threshold might give a different result
            # for fmr.
        logger.info(f"Plot FmrIapmr using: {input_names[1]}")
        if self._semilogx:
            mpl.semilogx(fmr_list, iapmr_list, label=label)
        else:
            mpl.plot(fmr_list, iapmr_list, label=label)

    def end_process(self):
        ''' Set title, legend, axis labels, grid colors, save figures and
        close pdf is needed '''
        # only for plots
        title = self._title if self._title is not None else "FMR vs IAPMR"
        if title.replace(' ', ''):
            mpl.title(title)
        mpl.xlabel(self._x_label or "FMR")
        mpl.ylabel(self._y_label or "IAPMR")
        mpl.grid(True, color=self._grid_color)
        if self._disp_legend:
            mpl.legend(loc=self._legend_loc)
        self._set_axis()
        fig = mpl.gcf()
        mpl.xticks(rotation=self._x_rotation)

        self._pdf_page.savefig(fig)

        # do not want to close PDF when running evaluate
        if 'PdfPages' in self._ctx.meta and \
                ('closef' not in self._ctx.meta or self._ctx.meta['closef']):
            self._pdf_page.close()
