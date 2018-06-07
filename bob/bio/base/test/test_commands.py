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
        assert result.exit_code == 0, (result.exit_code, result.output)
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
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', '-lg', 'A,B',
                               dev1, test1, dev2, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)
    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '-l', 'tmp', '-T', '0.1',
                               '--criterion', 'mindcf', '--cost', 0.9,
                               dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp',
                               '--criterion', 'mindcf', '--cost', 0.9,
                               dev1]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '--criterion', 'cllr', dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--criterion', 'cllr',
                               '--cost', 0.9, dev1]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-e', '--criterion', 'rr', '-T',
                               '0.1', dev1, test2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(
            commands.metrics, ['-l', 'tmp', '--criterion', 'rr',
                               dev1, dev2]
        )
        assert result.exit_code == 0, (result.exit_code, result.output)



def test_roc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output',
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
                                              'test.pdf', '-S', '-lc',
                                              'lower-left', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.roc, ['--output',
                                              'test.pdf',
                                              '-e', '--legends', 'A,B',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

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
        assert result.exit_code == 0, (result.exit_code, result.output)



def test_det():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/dev-4col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, [dev1, '-S'])
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
                                              'test.pdf', '--legends',
                                              'A,B', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    with runner.isolated_filesystem():
        result = runner.invoke(commands.det, ['--output',
                                              'test.pdf', '-e',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)


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
                                              '--legends', 'A,B', '-S',
                                              dev1, test1, dev2, test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

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
        result = runner.invoke(commands.hist, [dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion', 'min-hter', '--output',
                                               'HISTO.pdf', '-b',
                                               '30,auto', dev1, dev2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

    with runner.isolated_filesystem():
        result = runner.invoke(commands.hist, ['--criterion', 'eer', '--output',
                                               'HISTO.pdf', '-b', '30', '-e',
                                               '-lg', 'A,B', dev1, test1, dev2,
                                               test2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)



def test_cmc():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-cmc-5col.txt')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, [dev1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-cmc-4col.txt')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.cmc, ['--output', 'test.pdf',
                                              '--legends', 'A,B', '-S',
                                              '-ts', 'TA,TB', '-e',
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)

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
        assert result.exit_code == 0, (result.exit_code, result.output)




def test_dir():
    dev1 = pkg_resources.resource_filename('bob.bio.base.test',
                                           'data/scores-nonorm-openset-dev')
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dir, [dev1, '--rank', 2])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
    test1 = pkg_resources.resource_filename('bob.bio.base.test',
                                            'data/scores-nonorm-openset-dev')
    with runner.isolated_filesystem():
        result = runner.invoke(commands.dir, ['--output', 'test.pdf',
                                              '--legends', 'A,B', '-S',
                                              '--min-far-value', '1e-6', '-e',
                                              dev1, test1, dev1, test1])
        if result.output:
            click.echo(result.output)
        assert result.exit_code == 0, (result.exit_code, result.output)
