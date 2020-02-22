.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.experiments:


===========================================================================
Running Biometric Recognition Experiments: The Vanilla Biometrics in action
===========================================================================

Now, you are ready to run your first biometric recognition experiment.

.. _running_part_1:

Running Experiments.
--------------------

The previous section described the :ref:`bob.bio.base.struct_bio_rec_sys` using two sub-pipelines (three if you count the optional one) in a rough manner.
This section will describe in detail such sub-pipelines and its relation with biometric experiments.

These sub-pipelines were built using `Dask delayed <https://docs.dask.org/en/latest/delayed.html>`_ ; please follow the Dask documentation for more information about it.
Another source of information is the `TAM tutorial given at Idiap <https://github.com/tiagofrepereira2012/tam->`_


To run biometric experiments, we provide a generic CLI command called ``bob pipelines``.
Such CLI command is an entry-point to several pipelines implemented under `bob.pipelines`.
This tutorial will focus on the pipeline called `VANILLA-BIOMETRICS` (FIREWORKS PLEASE!!! BUM BUM BUM).

.. code-block:: sh

   bob pipelines vanilla-biometrics --help


By default, the ``vanilla-biometrics`` pipeline accepts one or more *configuration files* that include the parametrization of the experiment to run.
A configuration file contains one ore more *variables* that define parts of the experiment.
When several configuration files are specified, the variables of the latter will overwrite the ones of the former.
For simplicity, here we discuss only a single configuration file.

As a start, we have implemented a shortcut to generate an empty configuration file that contains all possible variables, each of which are documented and commented out:

.. code-block:: sh

   $ bob pipelines vanilla-biometrics-template experiment.py

.. note::
   The generated ``experiment.py`` is a regular python file, so you can include any regular python code inside this file.

Alright, lets have a look into this file.
Whoops, that's a lot of variables!
But, no worries, most of them have proper default values.
However, there are five variables, which are required and sufficient to define the complete biometric recognition experiment.
These five variables are:

* ``dask_client``: The Dask client pointing the execution backend
* ``database``: The database to run the experiments on
* ``preprocessor``: The data preprocessor
* ``extractor``: The feature extractor
* ``algorithm``: The recognition algorithm
* ``dask_client``: The Dask client pointing the execution backend


The first four variables, i.e., the ``database``, the ``preprocessor``, the ``extractor`` and the ``algorithm`` can be specified in several different ways.
For the start, we will use only the registered :ref:`Resources <bob.bio.base.resources>`.
These resources define the source code that will be used to compute the experiments, as well as all the meta-parameters of the algorithms (which we will call the *configuration*).
To get a list of registered resources, please call:

.. code-block:: sh

   $ resources.py

Each package in ``bob.bio`` defines its own resources, and the printed list of registered resources differs according to the installed packages.
If only ``bob.bio.base`` is installed, no databases and only one preprocessor will be listed.
To see more details about the resources, i.e., the full constructor call for the respective class, use the ``--details`` (or shortly ``-d``) option, and to sub-select only specific types of resources, use the ``--types`` (or ``-t``) option:

.. code-block:: sh

   $ resources.py -dt algorithm


.. Before going into :ref:`more details about the configurations <running_part_2>`, we will provide information about running default experiments.

One variable, which is not required, but recommended, is ``verbose``.
By default, the algorithms are set up to execute quietly, and only errors are reported (``logging.ERROR``).
To change this behavior, you can set the ``verbose`` variable to show:

1) Warning messages (``logging.WARN``)
2) Informative messages (``logging.INFO``)
3) Debug messages (``logging.DEBUG``)

When running experiments, my personal preference is verbosity level ``2``.
So, a minimal configuration file (say: ``pca_atnt.py``) would look something like:

.. code-block:: py

    from bob.bio.base.pipelines.blocks import DatabaseConnector, AlgorithmAdaptor
    import functools
    import bob.db.atnt

    database = 'atnt'

    preprocessor = 'face-detect'

    extractor = 'linearize'

    from bob.bio.base.algorithm import PCA
    algorithm = AlgorithmAdaptor(functools.partial(PCA, 0.99))


Running the experiment is then as simple as:

.. code-block:: sh

   $ bob pipelines vanilla-biometrics pca_atnt.py local_parallel.py

.. note::
   To be able to run exactly the command line from above, it requires to have :ref:`bob.bio.face <bob.bio.face>` installed.

.. note::
   The 'dask_client' variable is defined in the configuration file `local_parallel.py`. Check it out the package `bob.pipelines <http://gitlab.idiap.ch/bob/bob.pipelines>`_.


.. note::
   Chain loading is possible through configuration files, i.e., variables of each
   config is available during evaluation of the following config file.

   This allows us to spread our experiment setup in several configuration files and have a call similar to this::

   $ bob pipelines .py config_1.py config_2.py config_n.py

   For more information see *Chain Loading* in :ref:`bob.extension.config`.


The final result of the experiment will be one (or more) score file(s).
Usually, they will be called something like `scores-dev` in your output directory.


.. _bob.bio.base.command_line:

Command Line Options
--------------------
Each configuration can also directly be specified as command line option of ``bob pipelines vanilla-biometrics``.

.. note::
   Command line options have a long version starting with ``--`` and often a short version starting with a single ``-``.
   Here, only the long names of the arguments are listed, please refer to ``bob pipelines vanilla-biometrics --help`` (or short: ``bob pipelines vanilla-biometrics -h``) for the abbreviations.

Usually, the (long version of the) command line parameter is identical to the variable name, where ``_`` characters are replaced by ``-``, and all options start with ``--``.
Only, the ``--verbose`` option differs, you can use the ``--verbose`` option several times to increase verbosity, e.g, ``--verbose --verbose`` (or short ``-vv``) increases verbosity to ``2`` (alias ``logging.INFO``).
Generally, options defined on the command line will overwrite variables inside the configuration file(s).

.. note::
   Required options need to be specified at least in either the configuration file or on command line.
   If all options are given on the command line, the configuration file can be omitted completely.

The exact same experiment as above can, hence, be executed using:

.. code-block:: sh

   $ bob pipelines vanilla-biometrics --database mobio-image --preprocessor face-crop-eyes --extractor linearize --algorithm pca --output pca-experiment -vv

.. note::
   When running an experiment twice, you might realize that the second execution of the same experiment is much faster than the first one.
   This is due to the fact that those parts of the experiment, which have been successfully executed before (i.e., the according files already exist), are skipped.
   To override this behavior, i.e., to always regenerate all parts of the experiments, you can set ``force = True``.

However, to be consistent, throughout this documentation we document the options as variables.


.. _bob.bio.base.evaluate:

Evaluating Experiments
----------------------

After the experiment has finished successfully, one or more text file
containing all the scores are written. In this section, commands that helps to
quickly evaluate a set of scores by generating metrics or plots are presented
here. The scripts take as input either a 4-column or 5-column data format as
specified in the documentation of
:py:func:`bob.bio.base.score.load.four_column` or
:py:func:`bob.bio.base.score.load.five_column`.

Please note that there exists another file called ``Experiment.info`` inside
the result directory. This file is a pure text file and contains the complete
configuration of the experiment. With this configuration it is possible to
inspect all default parameters of the algorithms, and even to re-run the exact
same experiment.

Metrics
=======

To calculate the threshold using a certain criterion (EER (default), FAR or
min.HTER) on a development set and apply it on an evaluation set, just do:

.. code-block:: sh

    $ bob bio metrics -e {dev,test}-4col.txt --legends ExpA --criterion min-hter

    [Min. criterion: MIN-HTER ] Threshold on Development set `ExpA`: -4.830500e-03
    ======  ======================  =================
    ExpA    Development dev-4col    Eval. test-4col
    ======  ======================  =================
    FtA     0.0%                    0.0%
    FMR     6.7% (35/520)           2.5% (13/520)
    FNMR    6.7% (26/390)           6.2% (24/390)
    FAR     6.7%                    2.5%
    FRR     6.7%                    6.2%
    HTER    6.7%                    4.3%
    ======  ======================  =================

.. note::
    When evaluation scores are provided, ``--eval`` option must be passed.
    See metrics --help for further options.

You can also compute measure such as recognition rate (``rr``), Cllr and
minCllr (``cllr``) and minDCF (``mindcf``) by passing the corresponding option.
For example:

.. code-block:: sh

    bob bio metrics -e {dev,test}-4col.txt --legends ExpA --criterion cllr

    ======  ======================  ================
    Computing  Cllr and minCllr...
    =======  ======================  ================
    None     Development dev-4col    eval test-4col
    =======  ======================  ================
    Cllr     0.9%                    0.9%
    minCllr  0.2%                    0.2%
    =======  ======================  ================

.. note::
    You must provide files in the correct format depending on the measure you
    want to compute. For example, recognition rate takes cmc type files. See
    :py:func:`bob.bio.base.score.load.cmc`.

Plots
=====

Customizable plotting commands are available in the :py:mod:`bob.bio.base`
module. They take a list of development and/or evaluation files and generate a
single PDF file containing the plots. Available plots are:

*  ``roc`` (receiver operating characteristic)

*  ``det`` (detection error trade-off)

*  ``epc`` (expected performance curve)

*  ``hist`` (histograms of scores with threshold line)

*  ``cmc`` (cumulative match characteristic)

*  ``dir`` (detection & identification rate)

Use the ``--help`` option on the above-cited commands to find-out about more
options.


For example, to generate a CMC curve from development and evaluation datasets:

.. code-block:: sh

    $bob bio cmc -e -v --output 'my_cmc.pdf' dev-1.txt eval-1.txt
    dev-2.txt eval-2.txt

where `my_cmc.pdf` will contain CMC curves for the two experiments.

.. note::
    By default, ``det``, ``roc``, ``cmc`` and ``dir`` plot development and
    evaluation curves on
    different plots. You can force gather everything in the same plot using
    ``--no-split`` option.

.. note::
    The ``--figsize`` and ``--style`` options are two powerful options that can
    dramatically change the appearance of your figures. Try them! (e.g.
    ``--figsize 12,10 --style grayscale``)

Evaluate
========

A convenient command `evaluate` is provided to generate multiple metrics and
plots for a list of experiments. It generates two `metrics` outputs with EER,
HTER, Cllr, minDCF criteria along with `roc`, `det`, `epc`, `hist` plots for
each experiment. For example:

.. code-block:: sh

    $bob bio evaluate -e -v -l 'my_metrics.txt' -o 'my_plots.pdf' {sys1,sys2}/{dev,eval}

will output metrics and plots for the two experiments (dev and eval pairs) in
`my_metrics.txt` and `my_plots.pdf`, respectively.


.. include:: links.rst
