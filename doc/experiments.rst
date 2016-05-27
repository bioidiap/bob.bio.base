.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.experiments:


=========================================
Running Biometric Recognition Experiments
=========================================

Now, you are almost ready to run your first biometric recognition experiment.
Just a little bit of theory, and then: off we go.


Structure of a Biometric Recognition Experiment
-----------------------------------------------

Each biometric recognition experiment that is run with ``bob.bio`` is divided into several steps.
The steps are:

1. Data preprocessing: Raw data is preprocessed, e.g., for face recognition, faces are detected, images are aligned and photometrically enhanced.
2. Feature extractor training: Feature extraction parameters are learned.
3. Feature extraction: Features are extracted from the preprocessed data.
4. Feature projector training: Parameters of a subspace-projection of the features are learned.
5. Feature projection: The extracted features are projected into a subspace.
6. Model enroller training: The ways how to enroll models from extracted or projected features is learned.
7. Model enrollment: One model is enrolled from the features of one or more images.
8. Scoring: The verification scores between various models and probe features are computed.
9. Evaluation: The computed scores are evaluated and curves are plotted.

These 9 steps are divided into four distinct groups, which are discussed in more detail later:

* Preprocessing (only step 1)
* Feature extraction (steps 2 and 3)
* Biometric recognition (steps 4 to 8)
* Evaluation (step 9)

The communication between two steps is file-based, usually using a binary HDF5_ interface, which is implemented in the :py:class:`bob.io.base.HDF5File` class.
The output of one step usually serves as the input of the subsequent step(s).
Depending on the algorithm, some of the steps are not applicable/available.
E.g. most of the feature extractors do not need a special training step, or some algorithms do not require a subspace projection.
In these cases, the according steps are skipped.
``bob.bio`` takes care that always the correct files are forwarded to the subsequent steps.


.. _running_part_1:

Running Experiments (part I)
----------------------------

To run an experiment, we provide a generic script ``./bin/verify.py``, which is highly parametrizable.
To get a complete list of command line options, please run:

.. code-block:: sh

   $ ./bin/verify.py --help

Whoops, that's a lot of options.
But, no worries, most of them have proper default values.

.. note::
   Sometimes, command line options have a long version starting with ``--`` and a short one starting with a single ``-``.
   In this section, only the long names of the arguments are listed, please refer to ``./bin/verify.py --help`` (or short: ``./bin/faceverify.py -h``) for the abbreviations.

There are five command line options, which are required and sufficient to define the complete biometric recognition experiment.
These five options are:

* ``--database``: The database to run the experiments on
* ``--preprocessor``: The data preprocessor
* ``--extractor``: The feature extractor
* ``--algorithm``: The recognition algorithm
* ``--sub-directory``: A descriptive name for your experiment, which will serve as a sub-directory

The first four parameters, i.e., the ``database``, the ``preprocessor``, the ``extractor`` and the ``algorithm`` can be specified in several different ways.
For the start, we will use only the registered :ref:`Resources <bob.bio.base.resources>`.
These resources define the source code that will be used to compute the experiments, as well as all the meta-parameters of the algorithms (which we will call the *configuration*).
To get a list of registered resources, please call:

.. code-block:: sh

   $ ./bin/resources.py

Each package in ``bob.bio`` defines its own resources, and the printed list of registered resources differs according to the installed packages.
If only ``bob.bio.base`` is installed, no databases and only one preprocessor will be listed.
To see more details about the resources, i.e., the full constructor call fo the respective class, use the ``--details`` (or shortly ``-d``) option, and to sub-select only specific types of resources, use the ``--types`` (or ``-t``) option:

.. code-block:: sh

   $ ./bin/resources.py -dt algorithm


.. note::
   You will also find some ``grid`` resources being listed.
   These type of resources will be explained :ref:`later <running_in_parallel>`.

Before going into :ref:`more details about the configurations <running_part_2>`, we will provide information about running default experiments.

One command line option, which is not required, but recommended, is the ``--verbose`` option.
By default, the algorithms are set up to execute quietly, and only errors are reported.
To change this behavior, you can use the ``--verbose`` option several times to increase the verbosity level to show:

1) Warning messages
2) Informative messages
3) Debug messages

When running experiments, my personal preference is verbose level 2, which can be enabled by ``--verbose --verbose``, or using the short version: ``-vv``.
So, a typical biometric recognition experiment (in this case, face recognition) could look something like:

.. code-block:: sh

   $ ./bin/verify.py --database mobio-image --preprocessor face-crop-eyes --extractor linearize --algorithm pca --sub-directory pca-experiment -vv

.. note::
   To be able to run exactly the command line from above, it requires to have :ref:`bob.bio.face <bob.bio.face>` installed.

Before running an experiment, it is recommended to add the ``--dry-run`` option, so that it will only print, which steps would be executed, without actually executing them, and make sure that everything works as expected.

The final result of the experiment will be one (or more) score file(s).
Usually, they will be called something like ``scores-dev``.
By default, you can find them in a sub-directory the ``result`` directory, but you can change this option using the ``--result-directory`` command line option.

.. note::
   At Idiap_, the default result directory differs, see ``./bin/verify.py --help`` for your directory.


.. _bob.bio.base.evaluate:

Evaluating Experiments
----------------------

After the experiment has finished successfully, one or more text file containing all the scores are written.

To evaluate the experiment, you can use the generic ``./bin/evaluate.py`` script, which has properties for all prevalent evaluation types, such as CMC, ROC and DET plots, as well as computing recognition rates, EER/HTER, Cllr and minDCF.
Additionally, a combination of different algorithms can be plotted into the same files.
Just specify all the score files that you want to evaluate using the ``--dev-files`` option, and possible legends for the plots (in the same order) using the ``--legends`` option, and the according plots will be generated.
For example, to create a ROC curve for the experiment above, use:

.. code-block:: sh

   $ ./bin/evaluate.py --dev-files results/pca-experiment/male/nonorm/scores-dev --legend MOBIO --roc MOBIO_MALE_ROC.pdf -vv

Please note that there exists another file called ``Experiment.info`` inside the result directory.
This file is a pure text file and contains the complete configuration of the experiment.
With this configuration it is possible to inspect all default parameters of the algorithms, and even to re-run the exact same experiment.


.. _running_in_parallel:

Running in Parallel
-------------------

One important property of the ``./bin/verify.py`` script is that it can run in parallel, using either several threads on the local machine, or an SGE grid.
To achieve that, ``bob.bio`` is well-integrated with our SGE grid toolkit GridTK_, which we have selected as a python package in the :ref:`Installation <bob.bio.base.installation>` section.
The ``./bin/verify.py`` script can submit jobs either to the SGE grid, or to a local scheduler, keeping track of dependencies between the jobs.

The GridTK_ keeps a list of jobs in a local database, which by default is called ``submitted.sql3``, but which can be overwritten with the ``--gridtk-database-file`` option.
Please refer to the `GridTK documentation <http://pythonhosted.org/gridtk>`_ for more details on how to use the Job Manager ``./bin/jman``.

Two different types of ``grid`` resources are defined, which can be used with the ``--grid`` command line option.
The first type of resources will submit jobs to an SGE grid.
They are mainly designed to run in the Idiap_ SGE grid and might need some adaptations to run on your grid.
The second type of resources will submit jobs to a local queue, which needs to be run by hand (e.g., using ``./bin/jman --local run-scheduler --parallel 4``), or by using the command line option ``--run-local-scheduler``.
The difference between the two types of resources is that the local submission usually starts with ``local-``, while the SGE resource does not.

Hence, to run the same experiment as above using four parallel threads on the local machine, re-nicing the jobs to level 10, simply call:

.. code-block:: sh

   $ ./bin/verify.py --database mobio-image --preprocessor face-crop-eyes --extractor linearize --algorithm pca --sub-directory pca-experiment -vv --grid local-p4 --run-local-scheduler --nice 10

.. note::
   You might realize that the second execution of the same experiment is much faster than the first one.
   This is due to the fact that those parts of the experiment, which have been successfully executed before (i.e., the according files already exist), are skipped.
   To override this behavior, i.e., to always regenerate all parts of the experiments, you can use the ``--force`` option.


Command Line Options to change Default Behavior
-----------------------------------------------
Additionally to the required command line arguments discussed above, there are several options to modify the behavior of the experiments.
One set of command line options change the directory structure of the output.
By default, intermediate (temporary) files are by default written to the ``temp`` directory, which can be overridden by the ``--temp-directory`` command line option, which expects relative or absolute paths:

Re-using Parts of Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to re-use parts previous experiments, you can specify the directories (which are relative to the ``--temp-directory``, but you can also specify absolute paths):

* ``--preprocessed-data-directory``
* ``--extracted-directory``
* ``--projected-directory``
* ``--models-directories`` (one for each the models and the ZT-norm-models, see below)

or even trained extractor, projector, or enroller (i.e., the results of the extractor, projector, or enroller training):

* ``--extractor-file``
* ``--projector-file``
* ``--enroller-file``

For that purpose, it is also useful to skip parts of the tool chain.
To do that you can use:

* ``--skip-preprocessing``
* ``--skip-extractor-training``
* ``--skip-extraction``
* ``--skip-projector-training``
* ``--skip-projection``
* ``--skip-enroller-training``
* ``--skip-enrollment``
* ``--skip-score-computation``
* ``--skip-concatenation``
* ``--skip-calibration``

although by default files that already exist are not re-created.
You can use the ``--force`` argument combined with the ``--skip...`` arguments (in which case the skip is preferred).
To run just a sub-selection of the tool chain, you can also use the ``--execute-only`` option, which takes a list of options out of: ``preprocessing``, ``extractor-training``, ``extraction``, ``projector-training``, ``projection``, ``enroller-training``, ``enrollment``, ``score-computation``, ``concatenation`` or ``calibration``.


Database-dependent Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Many databases define several protocols that can be executed.
To change the protocol, you can either modify the configuration file, or simply use the ``--protocol`` option.

Some databases define several kinds of evaluation setups.
For example, often two groups of data are defined, a so-called *development set* and an *evaluation set*.
The scores of the two groups will be concatenated into two files called **scores-dev** and **scores-eval**, which are located in the score directory (see above).
In this case, by default only the development set is employed.
To use both groups, just specify ``--groups dev eval`` (of course, you can also only use the ``'eval'`` set by calling ``--groups eval``).

One score normalization technique is the so-called ZT score normalization.
To enable this, simply use the ``--zt-norm`` option.
If the ZT-norm is enabled, two sets of scores will be computed, and they will be placed in two different sub-directories of the score directory, which are by default called **nonorm** and **ztnorm**, but which can be changed using the ``--zt-score-directories`` option.


Other Arguments
---------------

Calibration
~~~~~~~~~~~

For some applications it is interesting to get calibrated scores.
Simply add the ``--calibrate-scores`` option and another set of score files will be created by training the score calibration on the scores of the ``'dev'`` group and execute it to all available groups.
The scores will be located at the same directory as the **nonorm** and **ztnorm** scores, and the file names are **calibrated-dev** (and **calibrated-eval** if applicable).

Unsuccessful Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, the preprocessor is not able to preprocess the data (e.g., for face image processing the face detector might not detect the face).
If you expect such cases to happen, you might want to use the ``--allow-missing-files`` option.
When this option is enabled, missing files will be handled correctly throughout the whole processing chain, i.e.:

* the data file is not used during training (in any step of the processing tool chain)
* preprocessed data is not written
* feature extraction is not performed for that file
* the file is exempt from model enrollment; if no enrollment file is found for a model, no model file is written
* if either model or probe file is not found, the according score will be ``NaN``.
  If several probe files are combined into one score, missing probe files will be ignored; if all probe files are not found, the score is ``NaN``.

.. warning::
   At the moment, combining the ``--allow-missing-files`` and ``zt-norm`` options might result in unexpected behavior, as the ZT-Norm computation does not handle ``NaN`` values appropriately.

.. include:: links.rst
