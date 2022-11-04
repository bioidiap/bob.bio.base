"""Runs error analysis on score sets, outputs metrics and plots"""

import logging

import click
import matplotlib.pyplot as mpl
import numpy as np

from tabulate import tabulate

import bob.measure.script.figure as measure_figure

from bob.measure import (
    f_score,
    far_threshold,
    farfrr,
    frr_threshold,
    min_weighted_error_rate_threshold,
    plot,
    ppndf,
    precision_recall,
    roc_auc_score,
)
from bob.measure.utils import get_thres, remove_nan

from . import error_utils

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


class Metrics(measure_figure.Metrics):
    """Compute metrics from score files

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
        super(Metrics, self).__init__(
            ctx, scores, evaluation, func_load, names, **kwargs
        )

    def _get_all_metrics(self, idx, input_scores, input_names):
        """Compute all metrics for dev and eval scores"""
        # Parse input and remove/count failed samples (NaN)
        dev_neg, dev_neg_na, dev_neg_count = remove_nan(
            input_scores[0]["licit_neg"]
        )
        dev_pos, dev_pos_na, dev_pos_count = remove_nan(
            input_scores[0]["licit_pos"]
        )
        dev_spoof, dev_spoof_na, dev_spoof_count = remove_nan(
            input_scores[0]["spoof"]
        )
        dev_fta = (dev_neg_na + dev_pos_na + dev_spoof_na) / (
            dev_neg_count + dev_pos_count + dev_spoof_count
        )
        if self._eval:
            eval_neg, eval_neg_na, eval_neg_count = remove_nan(
                input_scores[1]["licit_neg"]
            )
            eval_pos, eval_pos_na, eval_pos_count = remove_nan(
                input_scores[1]["licit_pos"]
            )
            eval_spoof, eval_spoof_na, eval_spoof_count = remove_nan(
                input_scores[1]["spoof"]
            )
            eval_fta = (eval_neg_na + eval_pos_na + eval_spoof_na) / (
                eval_neg_count + eval_pos_count + eval_spoof_count
            )
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
        res.append(
            self._strings(
                self._numbers(dev_neg, dev_pos, dev_spoof, threshold, dev_fta)
            )
        )

        if self._eval:
            # computes statistics for the eval set based on the threshold a
            # priori computed on the dev set
            res.append(
                self._strings(
                    self._numbers(
                        eval_neg, eval_pos, eval_spoof, threshold, eval_fta
                    )
                )
            )
        else:
            res.append(None)

        return res

    def _numbers(self, neg, pos, spoof, threshold, fta):
        """Computes each metric value"""
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
        iapmr, _ = farfrr(spoof, [0.0], threshold)
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
        """Formats the metrics values into strings"""
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
        """Compute metrics thresholds and tables for given system inputs"""
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
            rows.append([name, all_metrics[0].get(key, "N/A")])

        if self._eval:
            # computes statistics for the eval set based on the threshold a
            # priori
            headers.append("Evaluation")
            for row, key in zip(rows, self.names.keys()):
                row.append(all_metrics[1].get(key, "N/A"))

        click.echo(tabulate(rows, headers, self._tablefmt), file=self.log_file)


def _iapmr_dot(threshold, iapmr, real_data, **kwargs):
    # plot a dot on threshold versus IAPMR line and show IAPMR as a number
    axlim = mpl.axis()
    mpl.plot(threshold, 100.0 * iapmr, "o", color="C3", **kwargs)
    if not real_data:
        mpl.annotate(
            "IAPMR at\noperating point",
            xy=(threshold, 100.0 * iapmr),
            xycoords="data",
            xytext=(0.85, 0.6),
            textcoords="axes fraction",
            color="black",
            size="large",
            arrowprops=dict(facecolor="black", shrink=0.05, width=2),
            horizontalalignment="center",
            verticalalignment="top",
        )
    else:
        mpl.text(
            threshold + (threshold - axlim[0]) / 12,
            100.0 * iapmr,
            "%.1f%%" % (100.0 * iapmr,),
            color="C3",
        )


def _iapmr_line_plot(scores, n_points=100, **kwargs):
    axlim = mpl.axis()
    step = (axlim[1] - axlim[0]) / float(n_points)
    thres = [(k * step) + axlim[0] for k in range(2, n_points - 1)]
    mix_prob_y = []
    for k in thres:
        mix_prob_y.append(100.0 * error_utils.calc_pass_rate(k, scores))

    mpl.plot(thres, mix_prob_y, label="IAPMR", color="C3", **kwargs)


def _iapmr_plot(scores, threshold, iapmr, real_data, **kwargs):
    _iapmr_dot(threshold, iapmr, real_data, **kwargs)
    _iapmr_line_plot(scores, n_points=100, **kwargs)


class HistVuln(measure_figure.Hist):
    """Histograms for vulnerability"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(HistVuln, self).__init__(
            ctx, scores, evaluation, func_load, nhist_per_system=3
        )

    def _setup_hist(self, neg, pos):
        self._title_base = " "
        self._density_hist(pos[0], n=0, label="Genuine", color="C2")
        self._density_hist(
            neg[0], n=1, label="Zero-effort impostors", alpha=0.8, color="C0"
        )
        self._density_hist(
            neg[1],
            n=2,
            label="Presentation attack",
            alpha=0.4,
            color="C7",
            hatch="\\\\",
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
            eval_scores = {"licit_neg": [], "licit_pos": [], "spoof": []}

        threshold = (
            get_thres(
                self._criterion,
                dev_scores["licit_neg"],
                dev_scores["licit_pos"],
            )
            if self._thres is None
            else self._thres[idx]
        )
        return (
            [
                dev_scores["licit_neg"],
                dev_scores["spoof"],
            ],
            [dev_scores["licit_pos"]],
            [
                eval_scores["licit_neg"],
                eval_scores["spoof"],
            ],
            [
                eval_scores["licit_pos"],
            ],
            threshold,
        )

    def _lines(self, threshold, label, neg, pos, idx, **kwargs):
        spoof = neg[1]
        neg = neg[0]
        pos = pos[0]
        # plot EER treshold vertical line
        super(HistVuln, self)._lines(threshold, label, neg, pos, idx, **kwargs)

        if "iapmr_line" not in self._ctx.meta or self._ctx.meta["iapmr_line"]:
            # Plot iapmr_line (accepted PA vs threshold)
            iapmr, _ = farfrr(spoof, [0.0], threshold)
            ax2 = mpl.twinx()
            # we never want grid lines on axis 2
            ax2.grid(False)
            real_data = self._ctx.meta.get("real_data", True)
            _iapmr_plot(spoof, threshold, iapmr, real_data=real_data)
            n = idx % self._step_print
            col = n % self._ncols
            rest_print = (
                self.n_systems - int(idx / self._step_print) * self._step_print
            )
            if col == self._ncols - 1 or n == rest_print - 1:
                ax2.set_ylabel("IAPMR (%)", color="C3")
            ax2.tick_params(axis="y", colors="C3")
            ax2.yaxis.label.set_color("C3")
            ax2.spines["right"].set_color("C3")


class Epc(measure_figure.PlotBase):
    """Handles the plotting of EPC"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Epc, self).__init__(ctx, scores, evaluation, func_load)
        self._iapmr = self._ctx.meta.get("iapmr", True)
        self._titles = self._titles or [
            "EPC and IAPMR" if self._iapmr else "EPC"
        ]
        self._x_label = self._x_label or "Weight $\\beta$"
        self._y_label = self._y_label or "HTER (%)"
        self._eval = True  # always eval data with EPC
        self._split = False
        self._nb_figs = 1

        if self._min_arg != 2:
            raise click.BadParameter(
                "You must provide 2 scores files: " "scores-{dev,eval}.csv"
            )

    def compute(self, idx, input_scores, input_names):
        """Plot EPC with IAPMR for vuln"""
        dev_scores = clean_scores(input_scores[0])
        if self._eval:
            eval_scores = clean_scores(input_scores[1])
        else:
            eval_scores = {"licit_neg": [], "licit_pos": [], "spoof": []}

        mpl.gcf().clear()
        mpl.grid()
        logger.info(f"EPC using {input_names[0]} and {input_names[1]}")
        plot.epc(
            dev_scores["licit_neg"],
            dev_scores["licit_pos"],
            eval_scores["licit_neg"],
            eval_scores["licit_pos"],
            self._points,
            color="C0",
            linestyle=self._linestyles[idx],
            label=self._label("HTER (licit)", idx),
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
            apply_thres = [
                min_weighted_error_rate_threshold(
                    dev_scores["licit_neg"], dev_scores["licit_pos"], t
                )
                for t in thres
            ]
            mix_prob_y = []
            for k in apply_thres:
                mix_prob_y.append(
                    100.0 * error_utils.calc_pass_rate(k, eval_scores["spoof"])
                )

            logger.info(
                f"IAPMR in EPC plot using {input_names[0]}, {input_names[1]}"
            )
            mpl.plot(
                thres,
                mix_prob_y,
                label=self._label("IAPMR (spoof)", idx),
                color="C3",
            )

            prob_ax.tick_params(axis="y", colors="C3")
            prob_ax.yaxis.label.set_color("C3")
            prob_ax.spines["right"].set_color("C3")
            prob_ax.set_ylabel("IAPMR (%)", color="C3")
            prob_ax.set_axisbelow(True)
            ax1.yaxis.label.set_color("C0")
            ax1.tick_params(axis="y", colors="C0")
            ax1.spines["left"].set_color("C0")
            mpl.sca(ax1)


class Epsc(measure_figure.GridSubplot):
    """Handles the plotting of EPSC"""

    def __init__(self, ctx, scores, func_load, criteria, var_param, **kwargs):
        evaluation = ctx.meta.get("evaluation", True)
        super(Epsc, self).__init__(ctx, scores, evaluation, func_load)
        self._iapmr = self._ctx.meta.get("iapmr", False)
        self._wer = self._ctx.meta.get("wer", True)
        self._criteria = criteria or "eer"
        self._var_param = var_param or "omega"
        self._fixed_params = ctx.meta.get("fixed_params", [0.5])
        self._nb_subplots = 2 if (self._wer and self._iapmr) else 1
        if len(self._titles) < self._nb_figs * self._nb_subplots:
            self._titles = [
                v for v in self._titles for _ in range(self._nb_subplots)
            ]
        self._eval = True  # always eval data with EPSC
        self._split = False
        self._nb_figs = 1
        self._sampling = ctx.meta.get("sampling", 5)
        self._axis1 = None
        self._axis2 = None

        if self._min_arg != 2:
            raise click.BadParameter(
                "You must provide 2 scores files: " "scores-{dev,eval}.csv"
            )

        self._ncols = 1 if self._iapmr else 0
        self._ncols += 1 if self._wer else 0

    def compute(self, idx, input_scores, input_names):
        """Plot EPSC for vuln"""
        dev_scores = clean_scores(input_scores[0])
        if self._eval:
            eval_scores = clean_scores(input_scores[1])
        else:
            eval_scores = {"licit_neg": [], "licit_pos": [], "spoof": []}

        merge_sys = (
            self._fixed_params is None or len(self._fixed_params) == 1
        ) and self.n_systems > 1
        legend = ""
        if self._legends is not None and idx < len(self._legends):
            legend = self._legends[idx]
        elif self.n_systems > 1:
            legend = "Sys%d" % (idx + 1)

        if self._axis1 is None:
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
            if self._var_param == "omega":
                omega, beta, thrs = error_utils.epsc_thresholds(
                    dev_scores["licit_neg"],
                    dev_scores["licit_pos"],
                    dev_scores["spoof"],
                    dev_scores["licit_pos"],
                    points=points,
                    criteria=self._criteria,
                    beta=fp,
                )
            else:
                omega, beta, thrs = error_utils.epsc_thresholds(
                    dev_scores["licit_neg"],
                    dev_scores["licit_pos"],
                    dev_scores["spoof"],
                    dev_scores["licit_pos"],
                    points=points,
                    criteria=self._criteria,
                    omega=fp,
                )

            errors = error_utils.all_error_rates(
                eval_scores["licit_neg"],
                eval_scores["licit_pos"],
                eval_scores["spoof"],
                eval_scores["licit_pos"],
                thrs,
                omega,
                beta,
            )  # error rates are returned in a list in the
            # following order: frr, far, IAPMR, far_w, wer_w
            mpl.sca(self._axis1)
            # Between zero-effort impostors and Presentation attacks
            if self._wer:
                logger.debug(
                    f"Plotting EPSC: WER for system {idx+1}, fix param {pi}: {fp}"
                )
                set_title = (
                    self._titles[(idx // self.n_systems) * self._nb_subplots]
                    if self._titles
                    else None
                )
                display = (
                    set_title.replace(" ", "")
                    if set_title is not None
                    else True
                )
                wer_title = set_title or "EPSC"
                if display:
                    self._axis1.set_title(wer_title)
                base = f"({legend}) " if legend.strip() else ""
                if self._var_param == "omega":
                    label = f"{base}$\\beta={fp:.1f}$"
                    self._axis1.plot(
                        omega,
                        100.0 * errors[4].flatten(),
                        color=self._colors[pi],
                        linestyle="-",
                        label=label,
                    )
                    self._axis1.set_xlabel(self._x_label or "Weight $\\omega$")
                else:
                    label = f"{base}$\\omega={fp:.1f}$"
                    self._axis1.plot(
                        beta,
                        100.0 * errors[4].flatten(),
                        color=self._colors[pi],
                        linestyle="-",
                        label=label,
                    )
                    self._axis1.set_xlabel(self._x_label or "Weight $\\beta$")
                self._axis1.set_ylabel(
                    self._y_label or "WER$_{\\omega,\\beta}$ (%)"
                )
                self._axis1.grid(True)
                self._axis1.legend(loc=self._legend_loc)

            if self._iapmr:
                logger.debug(
                    f"Plotting EPSC: IAPMR for system {idx+1}, fix param {pi}: {fp}"
                )
                mpl.sca(self._axis2)
                set_title = (
                    self._titles[
                        (idx // self.n_systems) * self._nb_subplots + 1
                    ]
                    if self._titles
                    else None
                )
                display = (
                    set_title.replace(" ", "")
                    if set_title is not None
                    else True
                )
                iapmr_title = set_title or "EPSC"
                if display:
                    self._axis2.set_title(iapmr_title)
                base = f"({legend}) " if legend.strip() else ""
                if self._var_param == "omega":
                    label = f"{base} $\\beta={fp:.1f}$"
                    self._axis2.plot(
                        omega,
                        100.0 * errors[2].flatten(),
                        color=self._colors[pi],
                        linestyle="-",
                        label=label,
                    )
                    self._axis2.set_xlabel(self._x_label or "Weight $\\omega$")
                else:
                    label = f"{base} $\\omega={fp:.1f}$"
                    self._axis2.plot(
                        beta,
                        100.0 * errors[2].flatten(),
                        linestyle="-",
                        color=self._colors[pi],
                        label=label,
                    )
                    self._axis2.set_xlabel(self._x_label or "Weight $\\beta$")

                self._axis2.set_ylabel(self._y_label or "IAPMR (%)")
                self._axis2.grid(True)
                self._axis2.legend(loc=self._legend_loc)

    def end_process(self):
        """Sets the legend."""
        if self._end_setup_plot:
            for i in range(self._nb_figs):
                fig = mpl.figure(i + 1)
                if self._disp_legend:
                    mpl.legend(loc=self._legend_loc)
                self._pdf_page.savefig(fig)
        self._end_setup_plot = False
        super().end_process()


class Epsc3D(Epsc):
    """3D EPSC plots for vuln"""

    def __init__(self, ctx, scores, func_load, criteria, var_param, **kwargs):
        super().__init__(ctx, scores, func_load, criteria, var_param, **kwargs)
        if self._nb_subplots != 1:
            raise ValueError(
                "You cannot plot more than one type of plot (WER or IAPMR)."
            )

    def compute(self, idx, input_scores, input_names):
        """Implements plots"""
        dev_scores = clean_scores(input_scores[0])
        if self._eval:
            eval_scores = clean_scores(input_scores[1])
        else:
            eval_scores = {"licit_neg": [], "licit_pos": [], "spoof": []}

        default_title = "EPSC 3D"
        title = self._titles[idx] if self._titles else default_title

        mpl.rcParams.pop("key", None)

        points = self._sampling or 5

        # Compute threshold values on dev
        omega, beta, thrs = error_utils.epsc_thresholds(
            dev_scores["licit_neg"],
            dev_scores["licit_pos"],
            dev_scores["spoof"],
            dev_scores["licit_pos"],
            points=points,
            criteria=self._criteria,
        )

        # Compute errors on eval
        errors = error_utils.all_error_rates(
            eval_scores["licit_neg"],
            eval_scores["licit_pos"],
            eval_scores["spoof"],
            eval_scores["licit_pos"],
            thrs,
            omega,
            beta,
        )
        # error rates are returned in a list as 2D numpy.ndarrays in
        # the following order: frr, far, IAPMR, far_w, wer_wb, hter_wb
        wer_errors = 100 * errors[2 if self._iapmr else 4]

        if not self._axis1:
            self._axis1 = mpl.gcf().add_subplot(111, projection="3d")

        W, B = np.meshgrid(omega, beta)

        label = self._legends[idx] if self._legends else f"Sys {idx+1}"
        self._axis1.plot_wireframe(
            W,
            B,
            wer_errors,
            color=self._colors[idx],
            antialiased=False,
            label=label,
        )

        if self._iapmr:
            self._axis1.azim = -30
            self._axis1.elev = 50

        self._axis1.set_xlabel(self._x_label or r"Weight $\omega$")
        self._axis1.set_ylabel(self._y_label or r"Weight $\beta$")
        self._axis1.set_zlabel(
            r"WER$_{\omega,\beta}$ (%)" if self._wer else "IAPMR (%)"
        )

        if title.replace(" ", ""):
            mpl.title(title)


class BaseVulnDetRoc(measure_figure.PlotBase):
    """Base for DET and ROC"""

    def __init__(self, ctx, scores, evaluation, func_load, real_data, no_spoof):
        super(BaseVulnDetRoc, self).__init__(ctx, scores, evaluation, func_load)
        self._no_spoof = no_spoof
        self._fnmrs_at = ctx.meta.get("fnmr", [])
        self._fnmrs_at = [] if self._fnmrs_at is None else self._fnmrs_at
        self._real_data = True if real_data is None else real_data
        self._min_dig = -4 if self._min_dig is None else self._min_dig
        self._tpr = ctx.meta.get("tpr", True)

    def compute(self, idx, input_scores, input_names):
        """Implements plots"""
        dev_scores = clean_scores(input_scores[0])
        if self._eval:
            eval_scores = clean_scores(input_scores[1])
        else:
            eval_scores = {"licit_neg": [], "licit_pos": [], "spoof": []}

        mpl.figure(1)
        if self._eval:
            logger.info(f"dev curve using {input_names[0]}")
            self._plot(
                dev_scores["licit_neg"],
                dev_scores["licit_pos"],
                dev_scores["spoof"],
                npoints=self._points,
                tpr=self._tpr,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev", idx),
                alpha=self._alpha,
            )
            if not self._fnmrs_at:
                logger.info("Plotting fnmr line at dev eer threshold for dev")
                dev_threshold = get_thres(
                    criter="eer",
                    neg=dev_scores["licit_neg"],
                    pos=dev_scores["licit_pos"],
                )
                _, fnmr_at_dev_threshold = farfrr(
                    [0.0], dev_scores["licit_pos"], dev_threshold
                )
            fnmrs_dev = self._fnmrs_at or [fnmr_at_dev_threshold]
            self._draw_fnmrs(idx, dev_scores, fnmrs_dev)

            if self._split:
                mpl.figure(2)

            # Add the eval plot
            linestyle = "--" if not self._split else self._linestyles[idx]
            logger.info(f"eval curve using {input_names[1]}")
            self._plot(
                eval_scores["licit_neg"],
                eval_scores["licit_pos"],
                eval_scores["spoof"],
                linestyle=linestyle,
                npoints=self._points,
                tpr=self._tpr,
                min_far=self._min_dig,
                color=self._colors[idx],
                label=self._label("eval", idx),
                alpha=self._alpha,
            )
            if not self._fnmrs_at:
                logger.info("printing fnmr at dev eer threshold for eval")
                _, fnmr_at_dev_threshold = farfrr(
                    [0.0], eval_scores["licit_pos"], dev_threshold
                )
            fnmrs_dev = self._fnmrs_at or [fnmr_at_dev_threshold]
            self._draw_fnmrs(idx, eval_scores, fnmrs_dev, True)

        # Only dev scores available
        else:
            logger.info(f"dev curve using {input_names[0]}")
            self._plot(
                dev_scores["licit_neg"],
                dev_scores["licit_pos"],
                dev_scores["spoof"],
                npoints=self._points,
                tpr=self._tpr,
                min_far=self._min_dig,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev", idx),
                alpha=self._alpha,
            )
            if not self._fnmrs_at:
                logger.info("Plotting fnmr line at dev eer threshold for dev")
                dev_threshold = get_thres(
                    criter="eer",
                    neg=dev_scores["licit_neg"],
                    pos=dev_scores["licit_pos"],
                )
                _, fnmr_at_dev_threshold = farfrr(
                    [0.0], dev_scores["licit_pos"], dev_threshold
                )
            fnmrs_dev = self._fnmrs_at or [fnmr_at_dev_threshold]
            self._draw_fnmrs(idx, dev_scores, fnmrs_dev)

    def _get_farfrr(self, x, y, thres):
        return None, None

    def _plot(self, x, y, s, npoints, **kwargs):
        pass

    def _draw_fnmrs(self, idx, scores, fnmrs=[], eval=False):
        pass


class DetVuln(BaseVulnDetRoc):
    """DET for vuln"""

    def __init__(self, ctx, scores, evaluation, func_load, real_data, no_spoof):
        super(DetVuln, self).__init__(
            ctx, scores, evaluation, func_load, real_data, no_spoof
        )
        self._x_label = self._x_label or "FMR (%)"
        self._y_label = self._y_label or "FNMR (%)"
        self._semilogx = ctx.meta.get("semilogx", False)
        add = ""
        if not self._titles:
            self._titles = [""] * self._nb_figs
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        for i, t in enumerate(self._titles):
            if self._eval and (i % 2):
                dev_eval = ", eval group"
            elif self._eval:
                dev_eval = ", dev group"
            else:
                dev_eval = ""
            self._titles[i] = t or ("DET: LICIT" + add + dev_eval)
        self._legend_loc = self._legend_loc or "upper right"

    def _set_axis(self):
        if self._axlim is not None and None not in self._axlim:
            plot.det_axis(self._axlim)
        else:
            plot.det_axis([0.01, 99, 0.01, 99])

    def _get_farfrr(self, x, y, thres):
        points = farfrr(x, y, thres)
        return points, [ppndf(i) for i in points]

    def _plot(self, x, y, s, npoints, **kwargs):
        logger.info("Plotting DET")
        plot.det(
            x,
            y,
            npoints,
            min_far=self._min_dig,
            color=kwargs.get("color"),
            linestyle=kwargs.get("linestyle"),
            label=kwargs.get("label"),
        )
        if not self._no_spoof and s is not None:
            ax1 = mpl.gca()
            ax2 = ax1.twiny()
            ax2.set_xlabel("IAPMR (%)", color="C3")
            ax2.tick_params(
                axis="x",
                colors="C3",
                labelrotation=self._x_rotation,
                labelcolor="C3",
            )
            # Prevent tick labels overlap
            ax2.tick_params(axis="both", which="major", labelsize="x-small")
            ax1.tick_params(axis="both", which="major", labelsize="x-small")
            ax2.spines["top"].set_color("C3")
            plot.det(
                s,
                y,
                npoints,
                min_far=self._min_dig,
                color="C3",
                linestyle=":",
                label="Spoof " + kwargs.get("label"),
            )
            self._set_axis()
            mpl.sca(ax1)

    def _draw_fnmrs(self, idx, scores, fnmrs=[], eval=False):
        for line in fnmrs:
            thres_baseline = frr_threshold(
                scores["licit_neg"], scores["licit_pos"], line
            )

            axlim = mpl.axis()

            farfrr_licit, farfrr_licit_det = self._get_farfrr(
                scores["licit_neg"], scores["licit_pos"], thres_baseline
            )
            if farfrr_licit is None:
                return

            farfrr_spoof, farfrr_spoof_det = self._get_farfrr(
                scores["spoof"],
                scores["licit_pos"],
                frr_threshold(
                    scores["spoof"], scores["licit_pos"], farfrr_licit[1]
                ),
            )

            if not self._real_data:
                # Takes specified FNMR value as EER
                mpl.axhline(
                    y=farfrr_licit_det[1],
                    xmin=axlim[2],
                    xmax=axlim[3],
                    color="k",
                    linestyle="--",
                    label="%s @ EER" % self._y_label,
                )
            else:
                mpl.axhline(
                    y=farfrr_licit_det[1],
                    xmin=axlim[0],
                    xmax=axlim[1],
                    color="k",
                    linestyle="--",
                    label="%s = %.2f%%" % ("FNMR", farfrr_licit[1] * 100),
                )

            if not self._real_data:
                label_licit = "%s @ operating point" % self._x_label
                label_spoof = "IAPMR @ operating point"
            else:
                label_licit = "FMR=%.2f%%" % (farfrr_licit[0] * 100)
                label_spoof = "IAPMR=%.2f%%" % (farfrr_spoof[0] * 100)

            # Annotations and drawing of the points
            text_x_offset = 2
            text_y_offset = 5
            # Licit
            mpl.annotate(
                xy=(farfrr_licit_det[0], farfrr_licit_det[1]),
                text=label_licit,
                xytext=(text_x_offset, text_y_offset),
                textcoords="offset points",
                fontsize="small",
            )
            mpl.plot(
                farfrr_licit_det[0],
                farfrr_licit_det[1],
                "o",
                color=self._colors[idx],
            )  # FAR point, licit scenario
            # Spoof
            axlim = self._axlim or [0.01, 99, 0.1, 99]
            if (
                farfrr_spoof_det[0] > axlim[0]
                and farfrr_spoof_det[0] < axlim[1]
            ):
                mpl.annotate(
                    xy=(farfrr_spoof_det[0], farfrr_spoof_det[1]),
                    text=label_spoof,
                    xytext=(text_x_offset, text_y_offset),
                    textcoords="offset points",
                    fontsize="small",
                )
                mpl.plot(
                    farfrr_spoof_det[0],
                    farfrr_spoof_det[1],
                    "o",
                    color="C3",
                )  # FAR point, spoof scenario
            else:
                logger.warning(
                    f"The IAPMR for an FNMR of {line} is outside the plot."
                )


class RocVuln(BaseVulnDetRoc):
    """ROC for vuln"""

    def __init__(self, ctx, scores, evaluation, func_load, real_data, no_spoof):
        super(RocVuln, self).__init__(
            ctx, scores, evaluation, func_load, real_data, no_spoof
        )
        self._x_label = self._x_label or "FMR"
        self._y_label = self._y_label or "1 - FNMR"
        self._semilogx = ctx.meta.get("semilogx", True)
        add = ""
        if not self._titles:
            self._titles = [""] * self._nb_figs
        if not self._no_spoof:
            add = " and overlaid SPOOF scenario"
        for i, t in enumerate(self._titles):
            if self._eval and (i % 2):
                dev_eval = ", eval group"
            elif self._eval:
                dev_eval = ", dev group"
            else:
                dev_eval = ""
            self._titles[i] = t or ("ROC: LICIT" + add + dev_eval)
        if self._legend_loc == "best":
            self._legend_loc = (
                "lower right" if self._semilogx else "upper right"
            )

    def _plot(self, x, y, s, npoints, **kwargs):
        logger.info("Plotting ROC")
        plot.roc(
            x,
            y,
            npoints=npoints,
            semilogx=self._semilogx,
            tpr=self._tpr,
            min_far=self._min_dig,
            color=kwargs.get("color"),
            linestyle=kwargs.get("linestyle"),
            label=kwargs.get("label"),
        )
        if not self._no_spoof and s is not None:
            ax1 = mpl.gca()
            ax1.plot(
                [0],
                [0],
                linestyle=":",
                color="C3",
                label="Spoof " + kwargs.get("label"),
            )
            ax2 = ax1.twiny()
            ax2.set_xlabel("IAPMR (%)", color="C3")
            mpl.xticks(rotation=self._x_rotation)
            ax2.tick_params(
                axis="x",
                colors="C3",
                labelrotation=self._x_rotation,
                labelcolor="C3",
            )
            ax2.spines["top"].set_color("C3")
            plot.roc(
                s,
                y,
                npoints=npoints,
                semilogx=self._semilogx,
                tpr=self._tpr,
                min_far=self._min_dig,
                color="C3",
                linestyle=":",
                label="Spoof " + kwargs.get("label"),
            )
            self._set_axis()
            mpl.sca(ax1)

    def _get_farfrr(self, x, y, thres):
        points = farfrr(x, y, thres)
        points2 = (points[0], 1 - points[1])
        return points, points2

    def _draw_fnmrs(self, idx, scores, fnmrs=[], evaluation=False):
        for line in fnmrs:
            thres_baseline = frr_threshold(
                scores["licit_neg"], scores["licit_pos"], line
            )

            axlim = mpl.axis()

            farfrr_licit, farfrr_licit_roc = self._get_farfrr(
                scores["licit_neg"], scores["licit_pos"], thres_baseline
            )
            if farfrr_licit is None:
                return

            farfrr_spoof, farfrr_spoof_roc = self._get_farfrr(
                scores["spoof"],
                scores["licit_pos"],
                frr_threshold(
                    scores["spoof"], scores["licit_pos"], farfrr_licit[1]
                ),
            )

            if not self._real_data and not evaluation:
                mpl.axhline(
                    y=farfrr_licit_roc[1],
                    xmin=axlim[2],
                    xmax=axlim[3],
                    color="k",
                    linestyle="--",
                    label=f"{self._y_label} @ EER",
                )
            elif not evaluation:
                mpl.axhline(
                    y=farfrr_licit_roc[1],
                    xmin=axlim[0],
                    xmax=axlim[1],
                    color="k",
                    linestyle="--",
                    label=f"FNMR = {farfrr_licit[1] * 100:.2f}%",
                )

            if not self._real_data:
                label_licit = f"{self._x_label} @ operating point"
                label_spoof = "IAPMR @ operating point"
            else:
                label_licit = f"FMR={farfrr_licit[0] * 100:.2f}%"
                label_spoof = f"IAPMR={farfrr_spoof[0] * 100:.2f}%"

            mpl.plot(
                farfrr_licit_roc[0],
                farfrr_licit_roc[1],
                "o",
                color=self._colors[idx],
                label=label_licit,
            )  # FAR point, licit scenario
            mpl.plot(
                farfrr_spoof_roc[0],
                farfrr_spoof_roc[1],
                "o",
                color="C3",
                label=label_spoof,
            )  # FAR point, spoof scenario


class FmrIapmr(measure_figure.PlotBase):
    """FMR vs IAPMR"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(FmrIapmr, self).__init__(ctx, scores, evaluation, func_load)
        self._eval = True  # Always ask for eval data
        self._split = False
        self._nb_figs = 1
        self._semilogx = ctx.meta.get("semilogx", False)
        if not self._titles:
            self._titles = [""] * self._nb_figs
        for i, t in enumerate(self._titles):
            self._titles[i] = t or "FMR vs IAPMR"
        self._x_label = self._x_label or "FMR"
        self._y_label = self._y_label or "IAPMR"
        if self._min_arg != 2:
            raise click.BadParameter(
                "You must provide 2 scores files: " "scores-{dev,eval}.csv"
            )

    def compute(self, idx, input_scores, input_names):
        """Implements plots"""
        dev_scores = clean_scores(input_scores[0])
        if self._eval:
            eval_scores = clean_scores(input_scores[1])
        fmr_list = np.linspace(0, 1, 100)
        iapmr_list = []
        for i, fmr in enumerate(fmr_list):
            thr = far_threshold(
                dev_scores["licit_neg"], dev_scores["licit_pos"], fmr
            )
            iapmr_list.append(farfrr(eval_scores["spoof"], [0.0], thr)[0])
            # re-calculate fmr since threshold might give a different result
            # for fmr.
            fmr_list[i], _ = farfrr(eval_scores["licit_neg"], [0.0], thr)
        label = (
            self._legends[idx]
            if self._legends is not None
            else f"system {idx+1}"
        )
        logger.info(f"Plot FmrIapmr using: {input_names[1]}")
        if self._semilogx:
            mpl.semilogx(fmr_list, iapmr_list, label=label)
        else:
            mpl.plot(fmr_list, iapmr_list, label=label)
