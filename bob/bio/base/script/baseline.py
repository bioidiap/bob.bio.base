#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
A script to run biometric recognition baselines
"""


from .. import load_resource
import os
from ..baseline import get_available_databases, search_preprocessor
from bob.extension.scripts.click_helper import (
    verbosity_option, log_parameters)
import click
import tempfile
import logging

logger = logging.getLogger("bob.bio.base")


EPILOG = '''\b
Example:
    $ bob bio baseline eigenface atnt -vvv

which will run the eigenface baseline (from bob.bio.face) on the atnt
database.
'''


@click.command(context_settings={'ignore_unknown_options': True,
                                 'allow_extra_args': True}, epilog=EPILOG)
@click.argument('baseline', required=True)
@click.argument('database', required=True)
@click.option('--parallel-training', default='verify', show_default=True,
              type=click.Choice(('verify', 'gmm', 'isv', 'ivector')),
              help='Which script to use for training the algorithm. Some '
              'algorithms would train more efficiently using a different '
              'script.')
@verbosity_option()
@click.pass_context
def baseline(ctx, baseline, database, parallel_training, **kwargs):
    """Run a biometric recognition baseline.

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
    log_parameters(logger)

    # Triggering training for each baseline/database
    loaded_baseline = load_resource(
        baseline, 'baseline', package_prefix="bob.bio.")

    # find the compatible preprocessor for this database
    database_data = get_available_databases()[database]
    db = search_preprocessor(database, loaded_baseline.preprocessors.keys())
    preprocessor = loaded_baseline.preprocessors[db]

    # this is the default sub-directory that is used
    if "-T" in ctx.args or "--temp-directory" in ctx.args:
        sub_directory = os.path.join(database, baseline)
    else:
        sub_directory = baseline

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
        sub_directory=sub_directory,
        groups="', '".join(database_data["groups"]),
        verbose=ctx.meta['verbosity'],
    )

    if parallel_training == "verify":
        from .verify import main
    elif parallel_training == "gmm":
        from bob.bio.gmm.script.verify_gmm import main
    elif parallel_training == "isv":
        from bob.bio.gmm.script.verify_isv import main
    elif parallel_training == "ivector":
        from bob.bio.gmm.script.verify_ivector import main

    algorithm = loaded_baseline.algorithm
    if 'gmm' in algorithm and parallel_training != 'gmm':
        logger.warning("GMM algorithms can train faster using the "
                       "``--parallel-training gmm`` option.")
    if 'isv' in algorithm and parallel_training != 'isv':
        logger.warning("ISV algorithms can train faster using the "
                       "``--parallel-training isv`` option.")
    if 'ivector' in algorithm and parallel_training != 'ivector':
        logger.warning("ivector algorithms can train faster using the "
                       "``--parallel-training ivector`` option.")

    with tempfile.NamedTemporaryFile(mode='w+t', prefix='{}_'.format(baseline),
                                     suffix='.py', delete=False, dir='.') as f:
        f.write(config)
        f.flush()
        f.seek(0)
        main([f.name] + ctx.args)
        click.echo("You may want to delete `{}' after the experiments are "
                   "finished running.".format(f.name))
