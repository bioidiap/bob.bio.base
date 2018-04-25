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
    result = runner.invoke(commands.metrics, ['--no-evaluation', dev1])
    with runner.isolated_filesystem():
        with open('tmp', 'w') as f:
            f.write(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, [dev1, test1, dev2, test2]
        )
        with open('tmp', 'w') as f:
            f.write(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '-ts', 'A,B',
                               dev1, test1, dev2, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '-T', '0.1',
                               '--criterion', 'mindcf', '--cost', 0.9,
                               dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['--no-evaluation', '-l', 'tmp',
                               '--criterion', 'mindcf', '--cost', 0.9,
                               dev1]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['--criterion', 'cllr', dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['--no-evaluation', '-l', 'tmp', '--criterion', 'cllr',
                               '--cost', 0.9, dev1]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['--criterion', 'rr', '-T',
                               '0.1', dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['--no-evaluation', '-l', 'tmp', '--criterion', 'rr',
                               dev1, dev2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_roc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--no-evaluation', '--output',
                                              'test.pdf',dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--split', '--output',
                                              'test.pdf',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output',
                                              'test.pdf', '--titles', 'A,B', 
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)


def test_det():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--no-evaluation', dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    dev2 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-5col.txt')
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-4col.txt')
    test2 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/test-5col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--split', '--output',
                                              'test.pdf', '--titles', 'A,B',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--output',
                                              'test.pdf',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

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
        assert result.exit_code == 0, (result.exit_code, result.output)
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
        assert result.exit_code == 0, (result.exit_code, result.output)

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
        result = runner.invoke(commands.hist, ['--no-evaluation', dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion', 'hter', '--output',
                                               'HISTO.pdf', '-b',
                                               30,'--no-evaluation', dev1, dev2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion', 'eer', '--output',
                                               'HISTO.pdf', '-b', 30,
                                               '-ts', 'A,B', dev1, test1, dev2,
                                               test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_cmc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-cmc-5col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, ['--no-evaluation', dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-cmc-4col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, ['--output', 'test.pdf',
                                              '--titles', 'A,B',
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

def test_dir():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-nonorm-openset-dev')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dir, ['--no-evaluation', dev1, '--rank', 2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-nonorm-openset-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dir, ['--output', 'test.pdf',
                                              '--titles', 'A,B',
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

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
        result = runner.invoke(commands.evaluate, ['-l', 'tmp', '-f', 0.03,
                                                   '--no-evaluation', dev1, dev2])
        assert result.exit_code == 0, (result.exit_code, result.output)
        result = runner.invoke(commands.evaluate, ['--no-evaluation', '-f', 0.02,
                                                   dev1, dev2])
        assert result.exit_code == 0, (result.exit_code, result.output)

        result = runner.invoke(commands.evaluate, ['-l', 'tmp', '-f', 0.04,
                                                   dev1, test1, dev2, test2])
        assert result.exit_code == 0, (result.exit_code, result.output)
        result = runner.invoke(commands.evaluate, ['-f', 0.01,
                                                   dev1, test1, dev2, test2])
        assert result.exit_code == 0, (result.exit_code, result.output)

