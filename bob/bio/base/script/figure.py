"""Plots and measures for bob.bio.base"""

import logging
import math

import click
import matplotlib.pyplot as mpl

from tabulate import tabulate

import bob.measure
import bob.measure.script.figure as measure_figure

from bob.measure import plot

LOGGER = logging.getLogger("bob.bio.base")


class Roc(measure_figure.Roc):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Roc, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = ctx.meta.get("x_label") or "FMR"
        default_y_label = "1 - FNMR" if self._tpr else "FNMR"
        self._y_label = ctx.meta.get("y_label") or default_y_label


class Det(measure_figure.Det):
    def __init__(self, ctx, scores, evaluation, func_load):
        super(Det, self).__init__(ctx, scores, evaluation, func_load)
        self._x_label = ctx.meta.get("x_label") or "FMR (%)"
        self._y_label = ctx.meta.get("y_label") or "FNMR (%)"


class Cmc(measure_figure.PlotBase):
    """Handles the plotting of Cmc"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Cmc, self).__init__(ctx, scores, evaluation, func_load)
        self._semilogx = ctx.meta.get("semilogx", True)
        self._titles = self._titles or ["CMC dev.", "CMC eval."]
        self._x_label = self._x_label or "Rank"
        self._y_label = self._y_label or "Identification rate"
        self._max_R = 0

    def compute(self, idx, input_scores, input_names):
        """Plot CMC for dev and eval data using
        :py:func:`bob.measure.plot.cmc`"""
        mpl.figure(1)
        if self._eval:
            linestyle = "-" if not self._split else self._linestyles[idx]
            LOGGER.info("CMC dev. curve using %s", input_names[0])
            rank = plot.cmc(
                input_scores[0],
                logx=self._semilogx,
                color=self._colors[idx],
                linestyle=linestyle,
                label=self._label("dev.", idx),
            )
            self._max_R = max(rank, self._max_R)
            linestyle = "--"
            if self._split:
                mpl.figure(2)
                linestyle = self._linestyles[idx]

            LOGGER.info("CMC eval. curve using %s", input_names[1])
            rank = plot.cmc(
                input_scores[1],
                logx=self._semilogx,
                color=self._colors[idx],
                linestyle=linestyle,
                label=self._label("eval.", idx),
            )
            self._max_R = max(rank, self._max_R)
        else:
            LOGGER.info("CMC dev. curve using %s", input_names[0])
            rank = plot.cmc(
                input_scores[0],
                logx=self._semilogx,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev.", idx),
            )
            self._max_R = max(rank, self._max_R)


class Dir(measure_figure.PlotBase):
    """Handles the plotting of DIR curve"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(Dir, self).__init__(ctx, scores, evaluation, func_load)
        self._semilogx = ctx.meta.get("semilogx", True)
        self._rank = ctx.meta.get("rank", 1)
        self._titles = self._titles or ["DIR curve"] * 2
        self._x_label = self._x_label or "False Positive Identification Rate"
        self._y_label = self._y_label or "True Positive Identification Rate"

    def compute(self, idx, input_scores, input_names):
        """Plot DIR for dev and eval data using
        :py:func:`bob.measure.plot.detection_identification_curve`"""
        mpl.figure(1)
        if self._eval:
            linestyle = "-" if not self._split else self._linestyles[idx]
            LOGGER.info("DIR dev. curve using %s", input_names[0])
            plot.detection_identification_curve(
                input_scores[0],
                rank=self._rank,
                logx=self._semilogx,
                color=self._colors[idx],
                linestyle=linestyle,
                label=self._label("dev", idx),
            )
            linestyle = "--"
            if self._split:
                mpl.figure(2)
                linestyle = self._linestyles[idx]

            LOGGER.info("DIR eval. curve using %s", input_names[1])
            plot.detection_identification_curve(
                input_scores[1],
                rank=self._rank,
                logx=self._semilogx,
                color=self._colors[idx],
                linestyle=linestyle,
                label=self._label("eval", idx),
            )
        else:
            LOGGER.info("DIR dev. curve using %s", input_names[0])
            plot.detection_identification_curve(
                input_scores[0],
                rank=self._rank,
                logx=self._semilogx,
                color=self._colors[idx],
                linestyle=self._linestyles[idx],
                label=self._label("dev", idx),
            )

        if self._min_dig is not None:
            mpl.xlim(xmin=math.pow(10, self._min_dig))


class Metrics(measure_figure.Metrics):
    """Compute metrics from score files"""

    def __init__(
        self,
        ctx,
        scores,
        evaluation,
        func_load,
        names=(
            "Failure to Acquire",
            "False Match Rate",
            "False Non Match Rate",
            "False Accept Rate",
            "False Reject Rate",
            "Half Total Error Rate",
        ),
    ):
        super(Metrics, self).__init__(ctx, scores, evaluation, func_load, names)

    def init_process(self):
        if self._criterion == "rr":
            self._thres = (
                [None] * self.n_systems if self._thres is None else self._thres
            )

    def compute(self, idx, input_scores, input_names):
        """Compute metrics for the given criteria"""
        title = self._legends[idx] if self._legends is not None else None
        headers = ["" or title, "Dev. %s" % input_names[0]]
        if self._eval and input_scores[1] is not None:
            headers.append("eval % s" % input_names[1])
        if self._criterion == "rr":
            rr = bob.measure.recognition_rate(input_scores[0], self._thres[idx])
            dev_rr = "%.1f%%" % (100 * rr)
            raws = [["RR", dev_rr]]
            if self._eval and input_scores[1] is not None:
                rr = bob.measure.recognition_rate(
                    input_scores[1], self._thres[idx]
                )
                eval_rr = "%.1f%%" % (100 * rr)
                raws[0].append(eval_rr)
            click.echo(
                tabulate(raws, headers, self._tablefmt), file=self.log_file
            )
        elif self._criterion == "mindcf":
            if "cost" in self._ctx.meta:
                cost = self._ctx.meta.get("cost", 0.99)
            threshold = (
                bob.measure.min_weighted_error_rate_threshold(
                    input_scores[0][0], input_scores[0][1], cost
                )
                if self._thres is None
                else self._thres[idx]
            )
            if self._thres is None:
                click.echo(
                    "[minDCF - Cost:%f] Threshold on Development set `%s`: %e"
                    % (cost, input_names[0], threshold),
                    file=self.log_file,
                )
            else:
                click.echo(
                    "[minDCF] User defined Threshold: %e" % threshold,
                    file=self.log_file,
                )
            # apply threshold to development set
            far, frr = bob.measure.farfrr(
                input_scores[0][0], input_scores[0][1], threshold
            )
            dev_far_str = "%.1f%%" % (100 * far)
            dev_frr_str = "%.1f%%" % (100 * frr)
            dev_mindcf_str = "%.1f%%" % (
                (cost * far + (1 - cost) * frr) * 100.0
            )
            raws = [
                ["FAR", dev_far_str],
                ["FRR", dev_frr_str],
                ["minDCF", dev_mindcf_str],
            ]
            if self._eval and input_scores[1] is not None:
                # apply threshold to development set
                far, frr = bob.measure.farfrr(
                    input_scores[1][0], input_scores[1][1], threshold
                )
                eval_far_str = "%.1f%%" % (100 * far)
                eval_frr_str = "%.1f%%" % (100 * frr)
                eval_mindcf_str = "%.1f%%" % (
                    (cost * far + (1 - cost) * frr) * 100.0
                )
                raws[0].append(eval_far_str)
                raws[1].append(eval_frr_str)
                raws[2].append(eval_mindcf_str)
            click.echo(
                tabulate(raws, headers, self._tablefmt), file=self.log_file
            )
        elif self._criterion == "cllr":
            cllr = bob.measure.calibration.cllr(
                input_scores[0][0], input_scores[0][1]
            )
            min_cllr = bob.measure.calibration.min_cllr(
                input_scores[0][0], input_scores[0][1]
            )
            dev_cllr_str = "%.1f%%" % cllr
            dev_min_cllr_str = "%.1f%%" % min_cllr
            raws = [["Cllr", dev_cllr_str], ["minCllr", dev_min_cllr_str]]
            if self._eval and input_scores[1] is not None:
                cllr = bob.measure.calibration.cllr(
                    input_scores[1][0], input_scores[1][1]
                )
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
            title = self._legends[idx] if self._legends is not None else None
            all_metrics = self._get_all_metrics(idx, input_scores, input_names)
            headers = [" " or title, "Development"]
            rows = [
                [self.names[0], all_metrics[0][0]],
                [self.names[1], all_metrics[0][1]],
                [self.names[2], all_metrics[0][2]],
                [self.names[3], all_metrics[0][3]],
                [self.names[4], all_metrics[0][4]],
                [self.names[5], all_metrics[0][5]],
            ]

            if self._eval:
                # computes statistics for the eval set based on the threshold a
                # priori
                headers.append("Evaluation")
                rows[0].append(all_metrics[1][0])
                rows[1].append(all_metrics[1][1])
                rows[2].append(all_metrics[1][2])
                rows[3].append(all_metrics[1][3])
                rows[4].append(all_metrics[1][4])
                rows[5].append(all_metrics[1][5])

            click.echo(
                tabulate(rows, headers, self._tablefmt), file=self.log_file
            )


class MultiMetrics(measure_figure.MultiMetrics):
    """Compute metrics from score files"""

    def __init__(self, ctx, scores, evaluation, func_load):
        super(MultiMetrics, self).__init__(
            ctx,
            scores,
            evaluation,
            func_load,
            names=(
                "Failure to Acquire",
                "False Match Rate",
                "False Non Match Rate",
                "False Accept Rate",
                "False Reject Rate",
                "Half Total Error Rate",
            ),
        )


class Hist(measure_figure.Hist):
    """Histograms for biometric scores"""

    def _setup_hist(self, neg, pos):
        self._title_base = "Biometric scores"
        self._density_hist(pos[0], n=0, label="Genuines", alpha=0.9, color="C2")
        self._density_hist(
            neg[0], n=1, label="Zero-effort impostors", alpha=0.8, color="C0"
        )
