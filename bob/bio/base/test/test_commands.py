'''Tests for bob.measure scripts'''

import sys
import filecmp
import click
from click.testing import CliRunner
import pkg_resources
from ..script import commands

def test_metrics():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    result = runner.invoke(commands.metrics, [dev1])
    with runner.isolated_filesystem():
        with open('tmp', 'w') as f:
            f.write(result.output)
        assert result.exit_code == 0
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['--test', dev1, test1, dev2, test2]
        )
        with open('tmp', 'w') as f:
            f.write(result.output)
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--test', dev1, test1, dev2, test2]
        )
        assert result.exit_code == 0
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--test', dev1, test2]
        )
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '-t', '-T', '0.1',
                               '--criter', 'mindcf', '--cost', 0.9,
                               dev1, test2]
        )
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp',
                               '--criter', 'mindcf', '--cost', 0.9,
                               dev1]
        )
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-t', '--criter', 'cllr', dev1, test2]
        )
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--criter', 'cllr', '--cost', 0.9,
                               dev1]
        )
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-t', '--criter', 'rr', '-T',
                               '0.1', dev1, test2]
        )
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--criter', 'rr', dev1, dev2]
        )
        assert result.exit_code == 0


def test_roc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output','test.pdf',dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--test', '--split', '--output',
                                              'test.pdf',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--test', '--output',
                                              'test.pdf', '--titles', 'A,B', 
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0


def test_det():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, [dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--test', '--split', '--output',
                                              'test.pdf', '--titles', 'A,B',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--test', '--output',
                                              'test.pdf',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

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
        assert result.exit_code == 0
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.tar.gz')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.epc, ['--output', 'test.pdf',
                                              '--titles', 'A,B',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

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
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criter', 'hter', '--output',
                                               'HISTO.pdf', '-b', 30,
                                               dev1, dev2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criter', 'eer', '--output',
                                               'HISTO.pdf', '-b', 30, '-F',
                                               3, dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

def test_cmc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-cmc-5col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, [dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-cmc-4col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, ['--output', 'test.pdf', '-t',
                                              '--titles', 'A,B', '-F', 3,
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

def test_dic():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-nonorm-openset-dev')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dic, [dev1, '--rank', 2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-nonorm-openset-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dic, ['--output', 'test.pdf', '-t',
                                              '--titles', 'A,B', '-F', 3,
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0

def test_evaluate():
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
        result = runner.invoke(commands.evaluate, ['-l', 'tmp', '-f', 0.03, '-M',
                                                   '-x', '-m', dev1, dev2])
        assert result.exit_code == 0
        result = runner.invoke(commands.evaluate, ['-f', 0.02, '-M',
                                                   '-x', '-m', dev1, dev2])
        assert result.exit_code == 0

        result = runner.invoke(commands.evaluate, ['-l', 'tmp', '-f', 0.04, '-M',
                                                   '-x', '-m', '-t', dev1, test1,
                                                   dev2, test2])
        assert result.exit_code == 0
        result = runner.invoke(commands.evaluate, ['-f', 0.01, '-M', '-t',
                                                   '-x', '-m', dev1, test1, dev2,
                                                   test2])
        assert result.exit_code == 0

        result = runner.invoke(commands.evaluate, [dev1, dev2])
        assert result.exit_code == 0

        result = runner.invoke(commands.evaluate, ['-R', '-D', '-H', '-E',
                                                   '-o', 'PLOTS.pdf', dev1, dev2])
        assert result.exit_code == 0

        result = runner.invoke(commands.evaluate, ['-t', '-R', '-D', '-H', '-E',
                                                   '-o', 'PLOTS.pdf',
                                                   test1, dev1, test2, dev2])
        assert result.exit_code == 0

    cmc = pkg_resources.resource_filename('bob.bio.base.test',
                                          'data/scores-cmc-4col.txt')
    cmc2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-cmc-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.evaluate, ['-r', cmc])
        assert result.exit_code == 0
        result = runner.invoke(commands.evaluate, ['-r', '-t', cmc, cmc2])
        assert result.exit_code == 0
        result = runner.invoke(commands.evaluate, ['-C', '-t', cmc, cmc2])
        assert result.exit_code == 0
        result = runner.invoke(commands.evaluate, ['-C', cmc, cmc2])
        assert result.exit_code == 0

    cmc = pkg_resources.resource_filename('bob.bio.base.test',
                                          'data/scores-cmc-4col-open-set.txt')
    cmc2 = pkg_resources.resource_filename('bob.bio.base.test',
                                          'data/scores-nonorm-openset-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.evaluate, ['-O', cmc])
        assert result.exit_code == 0
        result = runner.invoke(commands.evaluate, ['-O', '-t', cmc, cmc2])
        assert result.exit_code == 0
