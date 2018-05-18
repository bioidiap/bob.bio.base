#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
A script to run biometric recognition baselines
"""


from .. import load_resource
from .verify import main as verify
from ..baseline import get_available_databases, search_preprocessor
from bob.extension.scripts.click_helper import verbosity_option
import click
import tempfile
import logging

logger = logging.getLogger("bob.bio.base")


@click.command(context_settings={'ignore_unknown_options': True,
                                 'allow_extra_args': True})
@click.argument('baseline', required=True)
@click.argument('database', required=True)
@verbosity_option()
@click.pass_context
def baseline(ctx, baseline, database):
    """Run a biometric recognition baseline.

    \b
    Example:
        $ bob bio baseline eigenface atnt -vvv

    which will run the eigenface baseline (from bob.bio.face) on the atnt
    database.

    \b
    Check out all baselines available by running:
    `resource.py --types baseline`
    and all available databases by running:
    `resource.py --types database`

    This script accepts parameters accepted by verify.py as well.
    See `verify.py --help` for the extra options that you can pass.

    Hint: pass `--grid demanding` to run the baseline on the SGE grid.

    Hint: pass `--temp-directory <dir>` to set the directory for temporary files

    Hint: pass `--result-directory <dir>` to set the directory for resulting score files

    """
    # Triggering training for each baseline/database
    loaded_baseline = load_resource(
        baseline, 'baseline', package_prefix="bob.bio.")

    # find the compatible preprocessor for this database
    database_data = get_available_databases()[database]
    db = search_preprocessor(database, loaded_baseline.preprocessors.keys())
    preprocessor = loaded_baseline.preprocessors[db]

    logger.debug('Database groups are %s', database_data["groups"])

    # call verify with newly generated config file. We will create a new config
    # file to allow people to use chain-loading and further modify the
    # baselines. See: https://gitlab.idiap.ch/bob/bob.bio.video/issues/12
    config = '''
preprocessor = '{preprocessor}'
extractor = '{extractor}'
algorithm = '{algorithm}'
database = '{database}'
sub_directory = '{sub_directory}'
groups = ['{groups}']
verbose = {verbose}
'''.format(
        preprocessor=preprocessor,
        extractor=loaded_baseline.extractor,
        algorithm=loaded_baseline.algorithm,
        database=database,
        sub_directory=baseline,
        groups="', '".join(database_data["groups"]),
        verbose=ctx.meta['verbosity'],
    )

    with tempfile.NamedTemporaryFile(mode='w+t', prefix='{}_'.format(baseline),
                                     suffix='.py', delete=False, dir='.') as f:
        f.write(config)
        f.flush()
        f.seek(0)
        verify([f.name] + ctx.args)
        click.echo("You may want to delete `{}' after the experiments are "
                   "finished running.".format(f.name))
