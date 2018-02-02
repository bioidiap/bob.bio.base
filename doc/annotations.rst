.. _bob.bio.base.annotations:

Annotating biometric databases
==============================

It is often required to annotate the biometric samples before running
experiments. This often happens in face biometrics where each face is detected
and location of landmarks on the face is saved prior to running experiments.

To facilitate the process of annotating a new database, this package provides
a command-line script:

.. code-block:: sh

    $ bob bio annotate --help

This script accepts two main parameters a database object that inherits from
:any:`bob.bio.base.database.BioDatabase` and an annotator object that inherits
from :any:`bob.bio.base.annotator.Base`. Please see the help message of the
script for more information.

The script can also be run in parallel using :ref:`gridtk`:

.. code-block:: sh

    $ jman submit --array 64 -- bob bio annotate /path/to/config.py --array 64

The number that is given to the ``--array`` options should match.
