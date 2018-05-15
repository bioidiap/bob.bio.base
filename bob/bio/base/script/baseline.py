#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

"""
This script runs some face recognition baselines under some face databases

Examples:

This command line will run the facenet from David Sandberg using the ATnT dataset:
  `bob bio baseline --baseline facenet_msceleba_inception_v1 --database atnt`
  
"""


import bob.bio.base
import bob.io.base
import os
from bob.bio.base.script.verify import main as verify
from bob.bio.base.baseline import get_available_databases
from bob.extension.scripts.click_helper import (
    verbosity_option, ConfigCommand, ResourceOption)
import click


@click.command(entry_point_group='bob.bio.config', cls=ConfigCommand)
@click.option('--database', '-d', required=True, cls=ResourceOption, help="Registered database. Check it out `resources.py --types database` for ready to be used databases")
@click.option('--baseline', '-b', required=True, cls=ResourceOption, help="Registered baseline. Check it out `resources.py --types baseline` for ready to be used baseline")
@click.option('--temp-dir', '-T', required=False, cls=ResourceOption, help="The directory for temporary files")
@click.option('--result-dir', '-R', required=False, cls=ResourceOption, help="The directory for resulting score files")
@click.option('--grid', '-g', help="Execute the algorithm in the SGE grid.", is_flag=True)
@click.option('--zt-norm', '-z', help="Enable the computation of ZT norms (if the database supports it).", is_flag=True)
@verbosity_option(cls=ResourceOption)

def baseline(baseline, database, temp_dir, result_dir, grid, zt_norm, **kwargs):
    """
    Run a biometric recognition baselines

    Check it out all baselines available by typing `resource.py --types baseline`

    """

    def search_preprocessor(key, keys):
        """
        Wrapper that searches for preprocessors for specific databases.
        If not found, the default preprocessor is returned
        """
        for k in keys:
            if key.startswith(k):
                return k
        else:
            return "default"

    # Triggering training for each baseline/database    
    loaded_baseline = bob.bio.base.load_resource(baseline, 'baseline', package_prefix="bob.bio.")

    # this is the default sub-directory that is used
    sub_directory = os.path.join(database, baseline)
    database_data = get_available_databases()[database]
    parameters = [
        '-p', loaded_baseline.preprocessors[search_preprocessor(database, loaded_baseline.preprocessors.keys())],
        '-e', loaded_baseline.extractor,
        '-d', database,
        '-a', loaded_baseline.algorithm,
        '-vvv',
        '--temp-directory', temp_dir,
        '--result-directory', result_dir,
        '--sub-directory', sub_directory
    ]
    
    parameters += ['--groups'] + database_data["groups"]

    if grid:
        parameters += ['-g', 'demanding']

    if zt_norm and 'has_zt' in database_data:
        parameters += ['--zt-norm']

    verify(parameters)
