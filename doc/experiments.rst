.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.experiments:


==========================================
Running Biometric Recognition Experiments
==========================================

Now, you are ready to run your first biometric recognition experiment.

.. _running_part_1:

Running Experiments (part I)
----------------------------
To run an experiment, we provide a generic script ``verify.py``.
As a default, ``verify.py`` accepts one or more *configuration files* that include the parametrization of the experiment to run.
A configuration file contains one ore more *variables* that define parts of the experiment.
When several configuration files are specified, the variables of the latter will overwrite the ones of the former.
For simplicity, here we discuss only a single configuration file.

As a start, we have implemented a shortcut to generate an empty configuration file that contains all possible variables, each of which are documented and commented out:

.. code-block:: sh

   $ verify.py --create-configuration-file experiment.py

.. note::
   The generated ``experiment.py`` is a regular python file, so you can include any regular python code inside this file.

Alright, lets have a look into this file.
Whoops, that's a lot of variables!
But, no worries, most of them have proper default values.
However, there are five variables, which are required and sufficient to define the complete biometric recognition experiment.
These five variables are:

* ``database``: The database to run the experiments on
* ``preprocessor``: The data preprocessor
* ``extractor``: The feature extractor
* ``algorithm``: The recognition algorithm
* ``sub_directory``: A descriptive name for your experiment, which will serve as a sub-directory

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

.. note::
   You will also find some ``grid`` resources being listed.
   These type of resources will be explained :ref:`later <running_in_parallel>`.

Before going into :ref:`more details about the configurations <running_part_2>`, we will provide information about running default experiments.

One variable, which is not required, but recommended, is ``verbose``.
By default, the algorithms are set up to execute quietly, and only errors are reported (``logging.ERROR``).
To change this behavior, you can set the ``verbose`` variable to show:

1) Warning messages (``logging.WARN``)
2) Informative messages (``logging.INFO``)
3) Debug messages (``logging.DEBUG``)

When running experiments, my personal preference is verbosity level ``2``.
So, a minimal configuration file (say: ``pca_atnt.py``) would look something like:

.. code-block:: py

   database = 'atnt'
   preprocessor = 'face-detect'
   extractor = 'linearize'
   algorithm = 'pca'
   sub_directory = 'PCA_ATNT'
   verbose = 2

Running the experiment is then as simple as:

.. code-block:: sh

   $ verify.py pca_atnt.py

.. note::
   To be able to run exactly the command line from above, it requires to have :ref:`bob.bio.face <bob.bio.face>` installed.


.. note::
   Chain loading is possible through configuration files, i.e., variables of each
   config is available during evaluation of the following config file.
   
   This allows us to spread our experiment setup in several configuration files and have a call similar to this::
   
   $ verify.py config_1.py config_2.py config_n.py
  
   For more information see *Chain Loading* in :ref:`bob.extension.config`.


Before running an experiment, it is recommended to add set the variable ``dry_run = True``, so that it will only print, which steps would be executed, without actually executing them, and make sure that everything works as expected.

The final result of the experiment will be one (or more) score file(s).
Usually, they will be called something like ``scores-dev``.
By default, you can find them in a sub-directory the ``result`` directory, but you can change this option using the ``result_directory`` variable.

.. note::
   At Idiap_, the default result directory differs, see ``verify.py --help`` for your directory.
   

.. _bob.bio.base.command_line:

Command Line Options
--------------------
Each configuration can also directly be specified as command line option of ``verify.py``.

.. note::
   Command line options have a long version starting with ``--`` and often a short version starting with a single ``-``.
   Here, only the long names of the arguments are listed, please refer to ``verify.py --help`` (or short: ``verify.py -h``) for the abbreviations.

Usually, the (long version of the) command line parameter is identical to the variable name, where ``_`` characters are replaced by ``-``, and all options start with ``--``.
For example, the ``sub_directory`` variable can also be set by the ``--sub-directory`` command line option.
Only, the ``--verbose`` option differs, you can use the ``--verbose`` option several times to increase verbosity, e.g, ``--verbose --verbose`` (or short ``-vv``) increases verbosity to ``2`` (alias ``logging.INFO``).
Generally, options defined on the command line will overwrite variables inside the configuration file(s).

.. note::
   Required options need to be specified at least in either the configuration file or on command line.
   If all options are given on the command line, the configuration file can be omitted completely.

The exact same experiment as above can, hence, be executed using:

.. code-block:: sh

   $ verify.py --database mobio-image --preprocessor face-crop-eyes --extractor linearize --algorithm pca --sub-directory pca-experiment -vv

.. note::
   When running an experiment twice, you might realize that the second execution of the same experiment is much faster than the first one.
   This is due to the fact that those parts of the experiment, which have been successfully executed before (i.e., the according files already exist), are skipped.
   To override this behavior, i.e., to always regenerate all parts of the experiments, you can set ``force = True``.

While we recommend to use a configuration file to declare your experiment, some variables might be faster to be changed on the command line, such as ``--dry-run``, ``--verbose``, ``--force`` (see above), ``--parallel N``, or ``--skip-...`` (see below).
However, to be consistent, throughout this documentation we document the options as variables.


.. _bob.bio.base.evaluate:

Evaluating Experiments
----------------------

After the experiment has finished successfully, one or more text file containing
all the scores are written. In this section, commands that helps to quickly
evaluate a set of scores by generating metrics or plots are presented here.
The scripts take as input either a 4-column or 5-column data format as specified
in the documentation of :py:func:`bob.bio.base.score.load.four_column` or 
:py:func:`bob.bio.base.score.load.five_column`.

Metrics
=======

To calculate the threshold using a certain criterion (EER (default), FAR or
min.HTER) on a development set and apply it on an evaluation set, just do:

.. code-block:: sh

    $ bob bio metrics {dev,test}-4col.txt --legends ExpA --criterion hter

    [Min. criterion: HTER ] Threshold on Development set `ExpA`: -4.830500e-03
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
    You can compute analysis on development set(s) only by passing option
    ``--no-evaluation``. See metrics --help for further options.

You can also compute measure such as recognition rate (``rr``), Cllr and
minCllr (``cllr``) and minDCF (``mindcf``) by passing the corresponding option.
For example:

.. code-block:: sh

    bob bio metrics {dev,test}-4col.txt --legends ExpA --criterion cllr

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

Customizable plotting commands are available in the :py:mod:`bob.bio.base` module.
They take a list of development and/or evaluation files and generate a single PDF
file containing the plots. Available plots are:

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

    $bob bio cmc --output 'my_cmc.pdf' dev-1.txt eval-1.txt 
    dev-2.txt eval-2.txt

where `my_cmc.pdf` will contain CMC curves for the two experiments.

.. note::
    By default, ``det``, ``roc``, ``cmc`` and ``dir`` plot development and
    evaluation curves on
    different plots. You can force gather everything in the same plot using
    ``--no-split`` option.

Evaluate
========

A convenient command `evaluate` is provided to generate multiple metrics and
plots for a list of experiments. It generates two `metrics` outputs with EER, 
HTER, Cllr, minDCF criteria along with `roc`, `det`, `epc`, `hist` plots for each
experiment. For example:

.. code-block:: sh

    $bob bio evaluate -l 'my_metrics.txt' -o 'my_plots.pdf' {sys1, sys2}/
    {eval,dev}

will output metrics and plots for the two experiments (dev and eval pairs) in
`my_metrics.txt` and `my_plots.pdf`, respectively.

Evaluate script (deprecated)
============================

After the experiment has finished successfully, one or more text file containing all the scores are written.
To evaluate the experiment, you can use the generic ``evaluate.py`` script, which has properties for all prevalent evaluation types, such as CMC, DIR, ROC and DET plots, as well as computing recognition rates, EER/HTER, Cllr and minDCF.
Additionally, a combination of different algorithms can be plotted into the same files.
Just specify all the score files that you want to evaluate using the ``--dev-files`` option, and possible legends for the plots (in the same order) using the ``--legends`` option, and the according plots will be generated.
For example, to create a ROC curve for the experiment above, use:

.. code-block:: sh

   $ evaluate.py --dev-files results/pca-experiment/male/nonorm/scores-dev --legend MOBIO --roc MOBIO_MALE_ROC.pdf -vv


Please note that there exists another file called ``Experiment.info`` inside the result directory.
This file is a pure text file and contains the complete configuration of the experiment.
With this configuration it is possible to inspect all default parameters of the algorithms, and even to re-run the exact same experiment.

.. _running_in_parallel:

Running in Parallel
-------------------

One important property of the ``verify.py`` script is that it can run in parallel, using either several processes on the local machine, or an SGE grid.
To achieve that, ``bob.bio`` is well-integrated with our SGE grid toolkit GridTK_, which we have selected as a python package in the :ref:`Installation <bob.bio.base.installation>` section.
The ``verify.py`` script can submit jobs either to the SGE grid, or to a local scheduler, keeping track of dependencies between the jobs.

The GridTK_ keeps a list of jobs in a local database, which by default is called ``submitted.sql3``, but which can be overwritten with the ``gridtk_database_file`` variable.
Please refer to the `GridTK documentation <https://www.idiap.ch/software/bob/docs/bob/gridtk/stable>`_ for more details on how to use the Job Manager ``jman``.

Two different types of ``grid`` resources are defined, which can be used with the ``grid`` variable.
The first type of resources will submit jobs to an SGE grid.
They are mainly designed to run in the Idiap_ SGE grid and might need some adaptations to run on your grid.
The second type of resources will submit jobs to a local queue, which needs to be run by hand (e.g., using ``jman --local run-scheduler --parallel 4``), or by setting the variable ``run_local_scheduler = True``.
The difference between the two types of resources is that the local submission usually starts with ``local-``, while the SGE resource does not.
You can also re-nice the parallel jobs by setting the ``nice`` variable accordingly.

To run an experiment parallel on the local machine, you can also use the simple variable ``parallel = N``, which will run the experiments in ``N`` parallel processes on your machine.
Here, ``N`` can be any positive integer -- but providing ``N`` greater than the number of processor threads of your machine will rather slow down processing.
Basically, ``parallel = N`` is a shortcut for:

.. code-block:: py

   grid = bob.bio.base.grid.Grid("local", number_of_parallel_processes=N)
   run_local_scheduler = True
   stop_on_failure = True

.. warning::
   Some of the processes require a lot of memory, which are multiplied by ``N`` when you run in ``N`` parallel processes.
   There is no check implemented to avoid that.


Variables to change Default Behavior
------------------------------------
Additionally to the required variables discussed above, there are several variables to modify the behavior of the experiments.
One set of command line options change the directory structure of the output.
By default, intermediate (temporary) files are by default written to the ``temp`` directory, which can be overridden by the ``temp_directory`` variable, which expects relative or absolute paths.

Re-using Parts of Experiments
=============================
If you want to re-use parts previous experiments, you can specify the directories (which are relative to the ``temp_directory``, but you can also specify absolute paths):

* ``preprocessed_directory``
* ``extracted_directory``
* ``projected_directory``
* ``models_directories`` (one for each the models and the ZT-norm-models, see below)

or even trained extractor, projector, or enroller (i.e., the results of the extractor, projector, or enroller training):

* ``extractor_file``
* ``projector_file``
* ``enroller_file``

For that purpose, it is also useful to skip parts of the tool chain.
To do that you can set these variables to ``True``:

* ``skip_preprocessing``
* ``skip_extractor_training``
* ``skip_extraction``
* ``skip_projector_training``
* ``skip_projection``
* ``skip_enroller_training``
* ``skip_enrollment``
* ``skip_score_computation``
* ``skip_concatenation``
* ``skip_calibration``

although by default files that already exist are not re-created.
You can use the ``force`` variable combined with the ``skip_`` variables (in which case the skip is preferred).
To (re-)run just a sub-selection of the tool chain, you can also use the ``execute_only`` variable, which takes a list of options out of: ``preprocessing``, ``extractor-training``, ``extraction``, ``projector-training``, ``projection``, ``enroller-training``, ``enrollment``, ``score-computation``, ``concatenation`` or ``calibration``.
This option is particularly useful for debugging purposes.


Database-dependent Variables
============================
Many databases define several protocols that can be executed.
To change the protocol, you can either modify the configuration file, or simply use the ``protocol`` variable.

Some databases define several kinds of evaluation setups.
For example, often two groups of data are defined, a so-called *development set* and an *evaluation set*.
The scores of the two groups will be concatenated into two files called **scores-dev** and **scores-eval**, which are located in the score directory (see above).
In this case, by default only the development set is employed.
To use both groups, just specify ``groups = ['dev', 'eval']`` (of course, you can also only use the ``'eval'`` set by setting ``groups = ['eval']``).

One score normalization technique is the so-called ZT score normalization.
To enable this, simply use the ``zt_norm`` variable.
If the ZT-norm is enabled, two sets of scores will be computed, and they will be placed in two different sub-directories of the score directory, which are by default called **nonorm** and **ztnorm**, but which can be changed using the ``zt_score_directories`` variable.


Other Variables
---------------

Calibration
===========
For some applications it is interesting to get calibrated scores.
Simply set the variable ``calibrate_scores = True`` and another set of score files will be created by training the score calibration on the scores of the ``'dev'`` group and execute it to all available groups.
The scores will be located at the same directory as the **nonorm** and **ztnorm** scores, and the file names are **calibrated-dev** (and **calibrated-eval** if applicable).

Unsuccessful Preprocessing
==========================
In some cases, the preprocessor is not able to preprocess the data (e.g., for face image processing the face detector might not detect the face).
If you expect such cases to happen, you might want to use the ``allow_missing_files`` variable.
When this variable is set to ``True``, missing files will be handled correctly throughout the whole processing chain, i.e.:

* the data file is not used during training (in any step of the processing tool chain)
* preprocessed data is not written
* feature extraction is not performed for that file
* the file is exempt from model enrollment; if no enrollment file is found for a model, no model file is written
* if either model or probe file is not found, the according score will be ``NaN``.
  If several probe files are combined into one score, missing probe files will be ignored; if all probe files are not found, the score is ``NaN``.

.. warning::
   At the moment, combining the ``allow_missing_files`` and ``zt_norm`` variables might result in unexpected behavior, as the ZT-Norm computation does not handle ``NaN`` values appropriately.

.. include:: links.rst
