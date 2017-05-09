.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.experiments:


===========================================
Running Biometric Verification Experiments
===========================================

Now, you are almost ready to run your first biometric verification experiment.
Just a little bit of theory, and then: off we go.

Structure of a Biometric Verification System
-----------------------------------------------

"Biometric verification" refers to the process of confirming that an invidual is who they say they are, 
based on their biometric data.  This implies that we have access to both the person's biometric data and 
their identity (e.g., a numerical ID, name, etc.).

A biometric verification system has two stages: 

1. **Enrollment:** A person's biometric data is added to the system's biometric database alongside the person's ID.
2. **Verification:** A person's biometric data is compared to the biometric data with the same ID in the system database, and a match score is generated.  The match score tells us how similar the two biometric samples are.  Based on a match threshold, we then decide whether or not the two biometric samples come from the same person (ID).

Fig. 1 shows the enrollment and verification stages in a typical biometric verification system:

.. figure:: /img/bio_ver_sys.svg
   :align: center

   Enrollment and verification in a typical biometric verification system.

* The "Pre-processor" cleans up the raw biometric data to make recognition easier (e.g., crops the face image to get rid of the background).
* The "Feature Extractor" extracts the most important features for recognition, from the pre-processed biometric data.
* The "Template Database" stores each person's extracted feature set (often referred to as a "template") along with their user ID.
* The "Matcher" compares a new biometric feature set to the template in the database that has the same user ID, and outputs a similarity score.
* The "Decision Maker" decides whether or not the new biometric sample and the template from the database match, based on whether the similarity score is above or below a pre-defined match threshold. 


Biometric Verification Experiments in bob.bio.base
--------------------------------------------------

In general, the goal of a biometric verification experiment is to quantify the verification accuracy of a biometric verification system, i.e., we wish to find out how good the system is at deciding whether a person is who they claim to be, based on their biometric data.

To conduct a biometric verification experiment, we need biometric data.  So, we use a biometric database.  A biometric database generally consists of multiple samples of a particular biometric, from multiple people.  For example, a face database could contain 5 different images of a person's face, from 100 people.  We then simulate "genuine" verification attempts by comparing each person's biometric samples to their other samples.  We simulate "impostor" verification attempts by comparing biometric samples across different people.

In bob.bio.base, biometric verification experiments are split up into four main stages, similar to the stages in a typical biometric verification system as illustrated in Fig. 1:

1. Data preprocessing
2. Feature extraction
3. Matching
4. Decision making

Each of these stages is discussed below:

*Data Preprocessing:*

Biometric measurements are often noisy, containing redundant information that is not necessary (and can be misleading) for verification.  For example, face images contain non-face background information, vein images can be unevenly illuminated, speech signals can be littered with background noise, etc.  The aim of the data preprocessing stage is to clean up the raw biometric data so that it is in the best possible state to make verification easier.  For example, biometric data is cropped from the background, the images are photometrically enhanced, etc.

All the biometric samples in the input biometric database go through the preprocessing stage.  The results are stored in a directory entitled "preprocessed".  This process is illustrated in Fig. 2:

.. figure:: /img/preprocessor.svg
   :align: center

   Preprocessing stage in bob.bio.base's biometric verification experiment framework.


*Feature Extraction:*

Although the preprocessing stage produces cleaner biometric data, the resulting data is usually very large and still contains much redundant information.  For example, only a few points in a person's face (e.g., eyes, nose, mouth, chin) are actually used for recognition purposes.  The aim of the feature extraction stage is to detect and extract only those features that are absolutely necessary for recognising a person.

All the biometric features stored in the "preprocessed" directory go through the feature extraction stage.  The results are stored in a directory entitled "extracted".  This process is illustrated in Fig. 3:

.. figure:: /img/extractor.svg
   :align: center

   Feature extraction stage in bob.bio.base's biometric verification experiment framework.

Note that there is sometimes a feature extractor training stage prior to the feature extraction (to help the extractor learn which features to extract), but this is not always the case.


*Matching:*

The matching stage in bob.bio.base is referred to as the "Algorithm".  Fig. 4 illustrates the Algorithm stage: 

.. figure:: /img/algorithm.svg
   :align: center

   Algorithm (matching) stage in bob.bio.base's biometric verification experiment framework.

From Fig. 4, we can see that the Algorithm stage consists of three main parts: 

(i) An optional "projection" stage after the feature extraction.  This would be used if, for example, you wished to project your extracted biometric features into a lower-dimensional subspace prior to verification.

(ii) Enrollment: The enrollment part of the Algorithm stage essentially works as follows.  One or more biometric samples per person is used to compute a representative "model" for that person.  This is the same as the "template" in our illustration of a typical biometric verification system in Fig. 1 and it essentially represents that person's identity.  To determine which of a person's biometric samples should be used to generate their model, we query our input biometric database.  The model is then calculated using the corresponding biometric features extracted in the Feature Extraction stage (or, optionally, our "projected" features).  Fig. 5 illustrates the enrollment part of the Algorithm module:

.. figure:: /img/algorithm_enrollment.svg
   :align: center

   The enrollment part of the Algorithm stage in bob.bio.base's biometric verification experiment framework.

Note that there is sometimes a model enroller training stage prior to enrollment.  This is only necessary when you are trying to fit an existing model to a set of biometric features, e.g., fitting a UBM to features extracted from a speech signal.  In other cases, the model is calculated from the features themselves, e.g., by averaging the feature vectors from multiple samples of the same biometric, in which case model enroller training is not necessary.


(iii) Scoring: The scoring part of the Algorithm stage essentially works as follows.  Each model is associated with a number of "probes".  In a biometric verification system, a probe is a biometric sample acquired during the verification stage (as opposed to the sample acquired during enrollment).  In the Scoring stage, we first query the input biometric database to determine which biometric samples should be used as the probes for each model.  Every model is then compared to its associated probes (some of which come from the same person, and some of which come from different people), and a score is calculated for each comparison.  The score may be a distance, and it tells us how similar or dissimilar the model and probe biometrics are.  Ideally, if the model and probe come from the same biometric (e.g., two images of the same finger), they should be very similar, and if they come from different biometrics (e.g., two images of different fingers) then they should be very different.  Fig. 6 illustrates the scoring part of the Algorithm module:

.. figure:: /img/algorithm_scoring.svg
   :align: center

   The scoring part of the Algorithm stage in bob.bio.base's biometric verification experiment framework.
 

*Decision Making:*

The decision making stage in bob.bio.base is referred to as "Evaluation".  The aim of this stage is to make a decision as to whether each score calculated in the Matching stage indicates a "Match" or "No Match" between the particular model and probe biometrics.  Once a decision has been made for each score, we can quantify the overall performance of the particular biometric verification system in terms of common metrics like the False Match Rate (FMR), False Non Match Rate (FNMR), and Equal Error Rate (EER).  We can also view a visual representation of the performance in terms of plots like the Receiver Operating Characteristic (ROC) and Detection Error Trade-off (DET).  Fig. 7 illustrates the Evaluation stage:

.. figure:: /img/evaluation.svg
   :align: center

   Evaluation stage in bob.bio.base's biometric verification experiment framework.


*Notes:*

* The communication between any two steps in the verification framework is file-based, usually using a binary HDF5_ interface, which is implemented in the :py:class:`bob.io.base.HDF5File` class.
* The output of one step usually serves as the input of the subsequent step(s), as portrayed in Fig. 2 -- Fig. 7.
* ``bob.bio`` ensures that the correct files are always forwarded to the subsequent steps.  For example, if you choose to implement a feature projection after the feature extraction stage, as illustrated in Fig. 4, ``bob.bio`` will make sure that the files in the "projected" directory are passed on as the input to the Enrollment stage; otherwise, the "extracted" directory will become the input to the Enrollment stage.


.. _running_part_1:

Running Experiments (part I)
----------------------------

To run an experiment, we provide a generic script ``verify.py``, which is highly parametrizable.
To get a complete list of command line options, please run:

.. code-block:: sh

   $ verify.py --help

Whoops, that's a lot of options.
But, no worries, most of them have proper default values.

.. note::
   Sometimes, command line options have a long version starting with ``--`` and a short one starting with a single ``-``.
   In this section, only the long names of the arguments are listed, please refer to ``verify.py --help`` (or short: ``faceverify.py -h``) for the abbreviations.

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

   $ resources.py

Each package in ``bob.bio`` defines its own resources, and the printed list of registered resources differs according to the installed packages.
If only ``bob.bio.base`` is installed, no databases and only one preprocessor will be listed.
To see more details about the resources, i.e., the full constructor call fo the respective class, use the ``--details`` (or shortly ``-d``) option, and to sub-select only specific types of resources, use the ``--types`` (or ``-t``) option:

.. code-block:: sh

   $ resources.py -dt algorithm


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

   $ verify.py --database mobio-image --preprocessor face-crop-eyes --extractor linearize --algorithm pca --sub-directory pca-experiment -vv

.. note::
   To be able to run exactly the command line from above, it requires to have :ref:`bob.bio.face <bob.bio.face>` installed.

Before running an experiment, it is recommended to add the ``--dry-run`` option, so that it will only print, which steps would be executed, without actually executing them, and make sure that everything works as expected.

The final result of the experiment will be one (or more) score file(s).
Usually, they will be called something like ``scores-dev``.
By default, you can find them in a sub-directory the ``result`` directory, but you can change this option using the ``--result-directory`` command line option.

.. note::
   At Idiap_, the default result directory differs, see ``verify.py --help`` for your directory.


.. _bob.bio.base.evaluate:

Evaluating Experiments
----------------------

After the experiment has finished successfully, one or more text file containing all the scores are written.

To evaluate the experiment, you can use the generic ``evaluate.py`` script, which has properties for all prevalent evaluation types, such as CMC, ROC and DET plots, as well as computing recognition rates, EER/HTER, Cllr and minDCF.
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

One important property of the ``verify.py`` script is that it can run in parallel, using either several threads on the local machine, or an SGE grid.
To achieve that, ``bob.bio`` is well-integrated with our SGE grid toolkit GridTK_, which we have selected as a python package in the :ref:`Installation <bob.bio.base.installation>` section.
The ``verify.py`` script can submit jobs either to the SGE grid, or to a local scheduler, keeping track of dependencies between the jobs.

The GridTK_ keeps a list of jobs in a local database, which by default is called ``submitted.sql3``, but which can be overwritten with the ``--gridtk-database-file`` option.
Please refer to the `GridTK documentation <http://pythonhosted.org/gridtk>`_ for more details on how to use the Job Manager ``jman``.

Two different types of ``grid`` resources are defined, which can be used with the ``--grid`` command line option.
The first type of resources will submit jobs to an SGE grid.
They are mainly designed to run in the Idiap_ SGE grid and might need some adaptations to run on your grid.
The second type of resources will submit jobs to a local queue, which needs to be run by hand (e.g., using ``jman --local run-scheduler --parallel 4``), or by using the command line option ``--run-local-scheduler``.
The difference between the two types of resources is that the local submission usually starts with ``local-``, while the SGE resource does not.

Hence, to run the same experiment as above using four parallel threads on the local machine, re-nicing the jobs to level 10, simply call:

.. code-block:: sh

   $ verify.py --database mobio-image --preprocessor face-crop-eyes --extractor linearize --algorithm pca --sub-directory pca-experiment -vv --grid local-p4 --run-local-scheduler --nice 10

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
