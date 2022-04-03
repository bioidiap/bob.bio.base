.. _bob.bio.base.annotators:

==============================
Annotating biometric databases
==============================

It is often required to annotate the biometric samples before running
experiments. This often happens in face biometrics where each face is detected
and the location of landmarks on the face is saved before running experiments.

To facilitate the process of annotating a new database, this package provides
a command-line script:

.. code-block:: sh

    $ bob bio annotate --help

This script accepts two main parameters a database object that inherits from
:any:`bob.bio.base.pipelines.Database` and an annotator
object that inherits from :any:`bob.bio.base.annotator.Annotator`. Please see
the help message of the script for more information.

The script can also be run in parallel using Dask:

.. code-block:: sh

    $ bob bio annotate /path/to/config.py --dask-client sge

You can find the list of readily available annotator configurations using the
``resources.py`` command:

.. code-block:: sh

    $ resources.py --types annotator
