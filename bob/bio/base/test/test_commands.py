'''Tests for bob.measure scripts'''

import click
from click.testing import CliRunner
import shutil
import pkg_resources
import numpy
import nose
from bob.extension.scripts.click_helper import assert_click_runner_result
from ..script import commands, sort, vuln_commands
from ..score import scores

def test_metrics():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    result = runner.invoke(commands.metrics, [dev1])
    with runner.isolated_filesystem():
        with open('tmp', 'w') as f:
            f.write(result.output)
        assert_click_runner_result(result)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', dev1, test1, dev2, test2]
        )
        with open('tmp', 'w') as f:
            f.write(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', '-lg', 'A,B',
                               dev1, test1, dev2, test2]
        )
        assert_click_runner_result(result)
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', dev1, test2]
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', '-T', '0.1',
                               '--criterion', 'mindcf', '--cost', 0.9,
                               dev1, test2]
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp',
                               '--criterion', 'mindcf', '--cost', 0.9,
                               dev1]
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '--criterion', 'cllr', dev1, test2]
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--criterion', 'cllr',
                               '--cost', 0.9, dev1]
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '--criterion', 'rr', '-T',
                               '0.1', dev1, test2]
        )
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--criterion', 'rr',
                               dev1, dev2]
        )
        assert_click_runner_result(result)


def test_roc():
    """

    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output',
                                              'test.pdf',dev1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--split', '--output',
                                              'test.pdf', '-S', '-ll',
                                              'lower-left', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output',
                                              'test.pdf',
                                              '-e', '--legends', 'A,B',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    dev_nonorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-nonorm-dev')
    dev_ztnorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-ztnorm-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, [
            '--min-far-value', '1e-6',
            '--lines-at', '1e-5',
            '-v', '--legends', 'A', '-e',
            dev_nonorm, dev_ztnorm
        ])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    """
    pass

def test_det():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, [dev1, '-S'])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--split', '--output',
                                              'test.pdf', '--legends',
                                              'A,B', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--output',
                                              'test.pdf', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    dev_nonorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-nonorm-dev')
    dev_ztnorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-ztnorm-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, [
            '--min-far-value', '1e-6',
            '--lines-at', '1e-5', '-e',
            '-v', '--legends', 'A',
            dev_nonorm, dev_ztnorm
        ])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_epc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, [dev1, test1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.tar.gz')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, ['--output', 'test.pdf',
                                              '--legends', 'A,B', '-S',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    dev_nonorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-nonorm-dev')
    dev_ztnorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-ztnorm-dev')

    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, [
            '-v', '--legends', 'A',
            dev_nonorm, dev_ztnorm
        ])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_hist():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, [dev1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion', 'min-hter', '--output',
                                               'HISTO.pdf', '-b',
                                               '30,auto', dev1, dev2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion', 'eer', '--output',
                                               'HISTO.pdf', '-b', '30', '-e',
                                               '-ts', 'A,B', dev1, test1, dev2,
                                               test2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_cmc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-cmc-5col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, [dev1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-cmc-4col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, ['--output', 'test.pdf',
                                              '--legends', 'A,B', '-S',
                                              '-ts', 'TA,TB', '-e',
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)

    dev_nonorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-nonorm-dev')
    dev_ztnorm = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/scores-ztnorm-dev')

    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, [
            '-v', '--legends', 'A', '-e',
            dev_nonorm, dev_ztnorm
        ])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_dir():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-nonorm-openset-dev')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dir, [dev1, '--rank', 2])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-nonorm-openset-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dir, ['--output', 'test.pdf',
                                              '--legends', 'A,B', '-S',
                                              '--min-far-value', '1e-6', '-e',
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert_click_runner_result(result)


def test_sort():

    def sorted_scores(score_lines):
        lines = []
        floats = []
        for line in score_lines:
            lines.append(line)
            floats.append(line[-1])
        sort_idx = numpy.argsort(floats)
        lines = [lines[i] for i in sort_idx]
        return lines

    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-nonorm-dev')
    runner = CliRunner()
    with runner.isolated_filesystem():
        # create a temporary sort file and sort it and check if it is sorted!

        path = "scores.txt"
        shutil.copy(dev1, path)

        result = runner.invoke(sort.sort, [path])
        assert_click_runner_result(result, exit_code=0)

        # load dev1 and sort it and compare to path
        dev1_sorted = sorted_scores(scores(dev1))
        path_scores = list(scores(path))

        nose.tools.assert_list_equal(dev1_sorted, path_scores)



def test_det_vuln():
    licit_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.det, ['-fnmr', '0.2',
                                                   '-o',
                                                   'DET.pdf',
                                                   licit_dev, licit_test,
                                                   spoof_dev, spoof_test])
        assert_click_runner_result(result)


def test_fmr_iapmr_vuln():
    licit_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.fmr_iapmr, [
            '--output', 'FMRIAPMR.pdf', licit_dev, licit_test, spoof_dev,
            spoof_test
        ])
        assert_click_runner_result(result)

        result = runner.invoke(vuln_commands.fmr_iapmr, [
            '--output', 'FMRIAPMR.pdf', licit_dev, licit_test, spoof_dev,
            spoof_test, '-G', '-L', '1e-7,1,0,1'
        ])
        assert_click_runner_result(result)


def test_hist_vuln():
    licit_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.hist,
                               ['--criterion', 'eer', '--output',
                                'HISTO.pdf', '-b', '30', '-ts', 'A,B',
                                licit_dev, licit_test])
        assert_click_runner_result(result)

    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.hist,
                               ['--criterion', 'eer', '--output',
                                'HISTO.pdf', '-b', '2,20,30', '-e',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)


def test_epc_vuln():
    licit_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.epc,
                               ['--output', 'epc.pdf',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)

        result = runner.invoke(vuln_commands.epc,
                               ['--output', 'epc.pdf', '-I',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)


def test_epsc_vuln():
    licit_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)

        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf', '-I',
                                '-fp', '0.1,0.3',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)

def test_epsc_3D_vuln():
    licit_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/licit/scores-dev')
    licit_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/licit/scores-eval')
    spoof_dev = pkg_resources.resource_filename('bob.bio.base.test',
                                                'data/spoof/scores-dev')
    spoof_test = pkg_resources.resource_filename('bob.bio.base.test',
                                                 'data/spoof/scores-eval')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf', '-D',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)

        result = runner.invoke(vuln_commands.epsc,
                               ['--output', 'epsc.pdf', '-D',
                                '-I', '--no-wer',
                                licit_dev, licit_test,
                                spoof_dev, spoof_test])
        assert_click_runner_result(result)
