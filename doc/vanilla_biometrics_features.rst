.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: Wed 24 Sep 2020 07:40:00 UTC+02

..  _bob.bio.base.vanilla_biometrics_advanced_features:

=====================================
Vanilla Biometrics: Advanced features
=====================================

There are several extra features that you can attach to the Vanilla Biometrics Pipeline.
In this section we'll explain the database interface, checkpointing of experiments, multitasking with Dask, and score file formats.

.. _bob.bio.base.database_interface:

The database interface
======================

A database interface is responsible for providing data samples to the pipeline when requested so.
It is the starting point of each sub-pipelines of vanilla-biometrics: Train, Enroll and Score.

.. note::

  Since the bob packages do not include the samples' data directly, the database interfaces need to be set up so that a path to the data is configurable.
  The interface only knows the file structure from that path and loads the data at runtime.

Groups of subjects are defined for the different parts of a biometric experiment.

  - The *world* set is used to train the model.
  - The *dev* set is used to enroll and score the subjects to define a working point for the system (setting a threshold value).
  - And for a complete experiment, the *eval* set is used to evaluate the performance of a biometric system on the working point given by the *dev* set scoring.

Each set must contain samples from individuals that are not represented in any other set.

Those set are then used in the experiment's pipeline:

  - The *world* set is passed to the training sub-pipeline to fit the transformers' background model.
  - Several samples (enough to enroll an individual) from each subject of the *dev* set are passed to the enroll sub-pipeline as references.
  - The rest of the samples of the *dev* set, the *probes*, are passed to the scoring sub-pipeline to generate the ``scores-dev`` file.
  - If needed, the last two steps are repeated for the *eval* set of subjects, to generate the ``scores-eval`` file.


Using a database interface
--------------------------

A series of ``bob.db`` interfaces already exist (find a list `here <https://pypi.org/search/?q=bob.db>`__). When such package is installed, you can list their resource names with::

$ resources.py -t d

You can use such a dataset with the following command (example with the AT&T dataset)::

$ bob bio pipelines vanilla-biometrics -d atnt -p <pipeline_name>

For more exotic datasets, you can simply pass your custom database file (defining a ``database`` object) to the vanilla-biometric pipeline::

$ bob bio pipelines vanilla-biometrics -d my_database.py -p <pipeline_name>

The ``database`` object defined in ``my_database.py`` is an instance of either:

- A :py:class:`~bob.bio.base.database.CSVDataset` (see :ref:`here <bob.bio.base.database.csv_file_interface>`),
- A :py:class:`~bob.bio.base.database.CSVDatasetCrossValidation` (see :ref:`here <bob.bio.base.database.csv_cross_validation>`),
- Your implementation of a :ref:`Database Interface <bob.bio.base.database.interface_class>`,
- A :ref:`legacy Database connector <bob.bio.base.legacy.database_connector>`.


.. _bob.bio.base.database.csv_file_interface:

CSV file Database interface
---------------------------

The easiest way to create a database interface is to use the CSV file interface.
This method is less complete and less flexible than implementing a :ref:`full interface class <bob.bio.base.database.interface_class>` but is sufficient for most use cases.

Protocol definition is possible and a set of CSV files (at least ``dev_enroll.csv`` and ``dev_probe.csv``) in a folder must be created for each protocol.

The interface is created with :py:class:`~bob.bio.base.database.CSVDataset`.
This class takes as input the base directory and the protocol sub-directory of the :ref:`CSV file structure <bob.bio.base.database.csv_file_structure>`, and finally, a ``csv_to_sample_loader`` that will load a sample data from a CSV row read from the CSV files.
This csv_to_sample_loader needs to know the dataset base path and the extension of the dataset files.

File format
^^^^^^^^^^^

You must provide a series of *comma separated values* (CSV) files containing at least two columns with a header:

.. code-block:: text

  PATH,REFERENCE_ID
  data/model11_session1_sample1,1
  data/model11_session1_sample2,1
  data/model11_session1_sample3,1
  data/model11_session2_sample1,1
  data/model12_session1_sample1,2
  data/model12_session1_sample2,2
  data/model12_session1_sample3,2
  data/model12_session2_sample1,2

The required columns in each file are the path to a sample (header: ``PATH``, relative to the dataset root) and a unique identifier for the individual represented by the sample (header: ``REFERENCE_ID``).

Metadata
........

This interface allows metadata to be shipped with the samples. To do so, simply add columns in the CSV file, with a corresponding header:

.. code-block:: text

  PATH,REFERENCE_ID,MY_METADATA_1,METADATA_2
  data/model11_session1_sample1,1,F,10
  data/model11_session1_sample2,1,F,10
  data/model11_session1_sample3,1,F,10
  data/model11_session2_sample1,1,F,10
  data/model12_session1_sample1,2,M,30
  data/model12_session1_sample2,2,M,30
  data/model12_session1_sample3,2,M,30
  data/model12_session2_sample1,2,M,30

.. _bob.bio.base.database.csv_file_structure:

File structure
^^^^^^^^^^^^^^

The following file structure and file naming must be followed, for the class to find the CSV files:

.. code-block:: text

  my_dataset
  |
  +-- my_protocol_1
  |   |
  |   +-- dev
  |      |
  |      +-- for_models
  |      +-- for_probes
  |
  +-- my_protocol_2
      |
      +-- norm
      |    |
      |    +-- train_world.csv
      |
      +-- dev
      |   |     
      |   +-- for_models.csv
      |   +-- for_probes.csv
      |
      +-- eval
           |
           +-- for_models.csv
           +-- for_probes.csv

- The minimal required files are the ``dev_enroll.csv`` and ``dev_probe.csv``, containing the sample paths and subjects of the *dev* set.
- The ``train.csv`` file (as shown in ``my_protocol_2``) is optional and contains the information of the *world* set.
- The ``eval_enroll.csv`` and ``eval_probe.csv`` files (as shown in ``my_protocol_2``) are optional and contain the information of the *eval* set.

In this example, ``my_dataset`` would be the base path given to the ``dataset_protocol_path`` parameter of :py:class:`~bob.bio.base.database.CSVDataset`, and ``my_protocol_1`` the ``protocol_name`` parameter:

.. code-block:: python

    from bob.bio.base.database import CSVDataset, AnnotationsLoader

    # Define a loading function called for each sample with its path
    def my_load_function(full_path):
        # Example with image samples
        import bob.io.image
        return bob.io.image.load(full_path)

    # Create a loader that takes the root of the dataset and a loader function
    my_sample_loader = CSVToSampleLoader(
        data_loader=my_load_function,
        dataset_original_directory="/path/to/dataset/root",
        extension=".png",
        metadata_loader=AnnotationsLoader()
    )

    # Create the CSV interface
    database = CSVDataset("my_dataset", "my_protocol_1", csv_to_sample_loader=my_sample_loader)

This will create a database interface with:

- The elements in ``train_world.csv`` returned by :py:meth:`~bob.db.base.Database.background_model_samples`,
- The elements in ``for_models.csv`` returned by :py:meth:`~bob.db.base.Database.references`,
- The elements in ``for_probes.csv`` returned by :py:meth:`~bob.db.base.Database.probes`.

An aggregation of all of the above is available with the :py:meth:`~bob.db.base.Database.all_samples` method, which returns all the samples of the protocol.

.. _bob.bio.base.database.csv_cross_validation:

CSV file Cross-validation Database interface
--------------------------------------------

The :py:class:`~bob.bio.base.database.CSVDatasetCrossValidation` takes only one CSV file of identities and creates the necessary sets pseudo-randomly.

The format of the CSV file is the same as in :py:class:`~bob.bio.base.database.CSVDataset`, comma separated with a header:

.. code-block:: text

  PATH,REFERENCE_ID
  path/to/sample0_subj0,0
  path/to/sample1_subj0,0
  path/to/sample2_subj0,0
  path/to/sample0_subj1,1
  path/to/sample1_subj1,1
  path/to/sample2_subj1,1

Two set are created: a *train* set and a *test* set. By default, the ratio between these two sets is defined at 0.2 *train* subjects and 0.8 *test* subjects.

By default, one sample of each subject of the *test* set will be used for enrollment. The rest of the samples will be used as probes against the models created by the enrolled samples.

To use the cross-validation database interface, use the following:

.. code-block:: python

    from bob.bio.base.database import CSVDatasetCrossValidation

    database = CSVDatasetCrossValidation(
        csv_file_name="your_dataset_name.csv",
        test_size=0.8,
        samples_for_enrollment=1,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load, 
            dataset_original_directory="", 
            extension="",
            metadata_loader=AnnotationsLoader()
        ),
    )

.. _bob.bio.base.database.interface_class:

The database interface class
----------------------------

Although most of the experiments will be satisfied with the CSV or cross-validation interfaces, there exists a way to specify exactly what is sent to each sub-pipelines by implementing your own database interface class.

When a vanilla-biometrics pipeline requests data from that class, it will call the following methods, which need to be implemented for each dataset:

  - :py:meth:`~bob.db.base.Database.background_model_samples`: Provides a list of :py:class:`~bob.pipelines.Sample` objects that are used for training of the :py:class:`Transformers`.
    Each :py:class:`~bob.pipelines.Sample` must contain at least the attributes :py:attr:`~bob.pipelines.Sample.key` and :py:attr:`~bob.pipelines.Sample.subject`, as well as the :py:attr:`~bob.pipelines.Sample.data` of the sample.
  - :py:meth:`~bob.db.base.Database.references`: Provides a list of :py:class:`~bob.pipelines.SampleSet` that are used for enrollment of the models.
    The group (*dev* or *eval*) can be given as parameter to specify which set must be used.
    Each :py:class:`~bob.pipelines.SampleSet` must contain a :py:attr:`~bob.pipelines.SampleSet.subject` attribute and a list of :py:attr:`~bob.pipelines.Sample` containing at least the :py:attr:`~bob.pipelines.Sample.key` attribute as well as the :py:attr:`~bob.pipelines.Sample.data` of the sample.
  - :py:meth:`~bob.db.base.Database.probes`: Returns a list of :py:class:`~bob.pipelines.SampleSet` that are used for scoring against a previously enrolled model.
    The group parameter (*dev* or *eval*) can be given to specify from which set of individuals the data comes.
    Each :py:class:`~bob.pipelines.SampleSet` must contain a :py:attr:`~bob.pipelines.SampleSet.subject`, a :py:attr:`~bob.pipelines.SampleSet.references` list, and a list of :py:attr:`~bob.pipelines.Sample` containing at least the :py:attr:`~bob.pipelines.Sample.key` attribute as well as the :py:attr:`~bob.pipelines.Sample.data` of the sample.

Furthermore, the :py:meth:`~bob.db.base.Database.all_samples` method must return a list of all the existing samples in the dataset. This functionality is used for annotating a whole dataset.

Here is a code snippet of a simple database interface:

.. code-block:: python

    from bob.pipelines import Sample, SampleSet

    class CustomDatabase:
        def background_model_samples(self):
            train_samples = []
            for a_sample in dataset_train_subjects:
                train_samples.append( Sample(data=a_sample.data, key=a_sample.sample_id) )
            return train_samples

        def references(self, group="dev"):
            all_references = []
            for a_subject in dataset_dev_subjects:
                current_sampleset = SampleSet(samples=[], reference_id=a_subject.id)
                for a_sample in a_subject:
                    current_sampleset.insert(-1, Sample(data=a_sample.data, key=a_sample.sample_id))
                all_references.append(current_sampleset)
            return all_references

        def probes(self, group="dev"):
            all_probes = []
            for a_subject in dataset_dev_subjects:
                current_sampleset = SampleSet(samples=[], reference_id=a_subject.id, references=list_of_references_id)
                for a_sample in a_subject:
                    current_sampleset.insert(-1, Sample(data=a_sample.data, key=a_sample.sample_id))
                all_probes.append(current_sampleset)
            return all_probes

        def all_samples(self, group=None):
            all_subjects = dataset_train_subjects + dataset_dev_subjects
            all_samples = []
            for a_sample in all_subjects:
                all_samples.append( Sample(data=a_sample.data, key=a_sample.sample_id) )
            return all_samples

        allow_scoring_with_all_biometric_references = True

    database = CustomDatabase()

.. note::

  For optimization reasons, an ``allow_scoring_with_all_biometric_references`` flag can be set in the database interface to allow scoring with all biometric references.
  This will be much faster when your algorithm allows vectorization of operations, but not all protocols allow such a feature.

  - When this flag is ``True``, the algorithm will compare a probe sample and generate the scores against **every** model of the set returned by the :py:meth:`references` method.
  - Otherwise (flag is ``False``), the scoring algorithm will only compare a probe to the given :py:attr:`~bob.pipelines.SampleSet.references` attribute of its :py:class:`~bob.pipelines.SampleSet`.


Delayed samples
^^^^^^^^^^^^^^^

To work with datasets too big to fit entirely in memory, the :py:class:`~bob.pipelines.DelayedSample` was introduced.

The functionality is the same as a :py:class:`~bob.pipelines.Sample`, but instead of storing the data as an attribute directly, a :py:meth:`~bob.pipelines.DelayedSample.load` method is used that can be set to any function loading one sample of data.
When data is needed, the load function is called and the sample data is returned.


Checkpointing experiments
=========================

Checkpoints are a useful tool that allows an experiment to prevent computing data multiple times by saving the results of each step so it can be retrieved later.
It can be used when an experiment fails in a later stage, preventing the computation of the stages coming before it, in case the experiment is restarted.

The checkpoints are files created on disk that contain the result of a sample of data passed through a :py:class:`Transformer` or :py:class:`BiometricAlgorithm`.
When running, if the system finds a checkpoint file for its current processing step, it will load the results directly from the disk, instead of computing it again.

To enable the checkpointing of a :py:class:`Transformer` or :py:class:`BiometricAlgorithm`, a :py:class:`~bob.pipelines.CheckpointWrapper` is available.
This class takes a :py:class:`Transformer` as input and returns the same :py:class:`Transformer` with the ability to automatically create checkpoint files.
The :py:class:`~bob.pipelines.CheckpointWrapper` class is available in the :py:mod:`bob.pipelines`.

The ``-c`` option (``--checkpoint``) is a command-line option that automatically wraps every steps of the pipeline with checkpointing::

$ bob bio pipelines vanilla-biometrics -d <database> -p <pipeline> -c -o <output_dir>

When doing so, the output of each :py:class:`Transformer` of the pipeline will be saved to the disk in the ``<output_dir>`` folder specified with the ``-o`` (``--output``) option.


.. WARNING::

  You have to be careful when using checkpoints: If you modify an early step of an experiment, the created checkpoints are not valid anymore, but the system has no way of knowing that.

  **You** have to take care of removing invalid checkpoints files.

  When changing the pipeline or the dataset of an experiment, you should change the output folder (``-o``) accordingly. Otherwise, the system could try to load a checkpoint of an older experiment, or samples from another dataset.


Scaling up with Dask
====================

`Dask <https://www.dask.org>`__ is a library that allows advanced parallelism of python programs.
This library takes care of creating jobs that can execute python code in parallel, schedules the execution of those jobs on the available resources, and manages the communication between those jobs.
Dask jobs can run in parallel on the same machine, or be spread on a *grid* infrastructure.
Diagnostic and monitoring tools are also available to watch progress and debug.


Using Dask with vanilla-biometrics
----------------------------------

To run an experiment with Dask, a :py:class:`bob.pipelines.DaskWrapper` class is available that takes any :py:class:`Transformer` or :py:class:`BiometricAlgorithm` and outputs a *dasked* version of it.

You can easily benefit from Dask by using the ``-l`` (``--dask-client``) option like so::

$ bob bio pipelines vanilla-biometrics -d <database> -p <pipeline> -l <client-config>

where ``<client-config>`` is your own dask client configuration file, or a resource from ``bob.pipelines.distributed``:

- ``local-parallel``: Launches as many jobs as :py:func:`multiprocessing.cpu_count` that will run simultaneously on the current machine.
- ``sge``: Runs jobs on a grid architecture, on any workers (at Idiap, uses the SGE. Other grids may require more configuration, see :py:mod:`bob.pipelines`).
- ``sge-gpu``: Executes jobs on a grid architecture, but defaults to the nodes with a GPU (q_short_gpu).


.. note::

  If the dask option is not set (omitting the ``-l``), everything will run locally in one single thread.


.. warning::

  **For Idiap users:** If you need to run the vanilla-biometrics in the SGE. Don't forget to do::

  $ SETSHELL grid

  Also, since the grid nodes are not allowed to create additional jobs on the grid, you cannot run the main dask client on a job (``qsub -- bob bio pipelines vanilla-biometrics -l sge_...`` will not work).


Monitoring and diagnostic
-------------------------

A running instance of Dask will create a server with a listening port (default: port 8787) that answers to HTTP requests.
This means that when an instance of Dask is running, you can open `localhost:8787 <http://localhost:8787>`_ in your browser to monitor the running jobs.

The page available contains some useful tools like a graphical visualization of the jobs and their dependencies, a resources monitor (CPU and memory of each node), and access to the logs of all running and finished jobs.

.. figure:: img/dask_monitor_status.png
  :figwidth: 75%
  :align: center
  :alt: Bar graphs showing different processes being executed at the same time.

  The *Status* page of the Dask monitoring tool. 8 jobs are running in parallel with different steps being processed.

.. figure:: img/dask_monitor_workers.png
  :figwidth: 75%
  :align: center
  :alt: A list of the workers with cpu and memory usages.

  The *Workers* page of the Dask monitoring tool. 8 workers are accepting jobs, and their current memory and CPU usage are plotted.

.. figure:: img/dask_monitor_graph.png
  :figwidth: 75%
  :align: center
  :alt: A graph of jobs and their dependencies.

  The *Graph* page of the Dask monitoring tool. We can observe the different steps of a biometric experiment, with the bottom part being the enrollment of references and the top part the probes used for scoring.
  The input are on the left side, and the resulting scores are generated by the jobs on the right.


Writing scores in a customized manner
=====================================

The results of the Vanilla-Biometrics pipelines are scores that can be analyzed.
To do so, the best way is to store them on disk so that a utility like ``bob bio metrics`` or ``bob bio roc`` can retrieve them.
However, many formats could be used to store those score files, so :py:class:`ScoreWriter` was defined.

A :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.ScoreWriter` must implement a :py:meth:`~bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.ScoreWriter.write` method that receives a list of :py:class:`~bob.pipelines.SampleSet` that contains scores as well as metadata (subject identity, sample path, etc.).

Common :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.ScoreWriter` are available by default:

- A CSV format ScoreWriter
- A four columns format ScoreWriter


Using a Score Writer
--------------------

By default (omitting the ``-m`` option), vanilla-biometrics will use the four-column format ScoreWriter.

To indicate to a vanilla-biometrics pipeline to use the CSV ScoreWriter instead of the default four-column ScoreWriter, you can pass the ``-m`` (``--write-metadata-scores``) option like so::

$ bob bio pipelines vanilla-biometrics -m -d <database> -p <pipeline> -o <output_dir>


CSV Score Writer
----------------

A :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.CSVScoreWriter` is available, creating a Comma Separated Values (CSV) file with all the available metadata of the dataset as well as the score of each comparison.
It is more complete than the :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.FourColumnsScoreWriter` and allows analysis of scores according to sample metadata (useful to analyze bias due to e.g. age or gender).

The default :py:class:`~bob.bio.base.pipelines.vanilla_biometrics.CSVScoreWriter` will write a *probe* subject and key, a *biometric reference* subject, the score resulting from the comparison of that *probe* against the *reference*, as well as all the fields set in :py:class:`~bob.pipelines.SampleSet` by the :ref:`database interface<bob.bio.base.database_interface>`, for the *references* and the *probes*.

A header is present, to identify the metadata, and the values are comma separated.
Here is a short example of such file format with metadata on the ``age`` and ``gender`` of the subject:

.. code-block:: none

  probe_subject, probe_key, probe_age, probe_gender, bio_ref_subject, bio_ref_age, bio_ref_gender, score
  subj_0, subj0_sample0, 27, male, subj_0, 29, male, 0.966
  subj_0, subj0_sample0, 29, male, subj_1, 45, female, 0.127
  subj_1, subj1_sample0, 45, female, subj_2, 40, female, 0.254
  subj_1, subj1_sample1, 46, female, subj_2, 40, female, 0.287


Four Columns Score Writer
-------------------------

:py:class:`~bob.bio.base.pipelines.vanilla_biometrics.FourColumnsScoreWriter` is the default score file format used by bob.
It consists of a text file with four columns separated by spaces and no header.
Each row represents a comparison between a *probe* and a *model*, and the similarity score resulting from that comparison.
The four columns are, in order:

1) **Reference ID**: The identity of the *reference* or *model* used for that comparison (field ``subject`` in :py:class:`~bob.pipelines.SampleSet` of the ``references`` set).
2) **Probe ID**: The identity of the *probe* used for that comparison (field ``subject`` in :py:class:`~bob.pipelines.SampleSet` of the ``probes`` set).
3) **Probe key**: A unique identifier (often a file path) for the sample used as *probe* for that comparison.
4) **Score**: The similarity score of that comparison.

Here is a short example of such file format:

.. code-block:: none

  subj_0 subj_0 subj0_sample0 0.998
  subj_0 subj_0 subj0_sample1 0.985
  subj_0 subj_1 subj1_sample0 0.450
  subj_0 subj_1 subj1_sample1 0.289


Evaluation
==========

Once scores are generated for each probe evaluated against the corresponding models, conclusions on the performance of the system can be drawn by using metrics like the FMR and FNMR or tools like the ROC curve.


In this section, commands that help to quickly evaluate a set of scores by generating metrics or plots are presented.
The scripts take as input a four-column file data format as specified in the documentation of :py:func:`bob.bio.base.score.load.four_column`.

.. note::

  If both *dev* and *eval* scores are available, the scores of the *dev* set will be used to define the threshold used to evaluate the scores of the *eval* set.

Metrics
-------

To calculate the threshold using a certain criterion (EER (default), FAR, or
min.HTER) on a development set and apply it on an evaluation set, just do:

.. code-block:: sh

  $ bob bio metrics -e output/scores-{dev,eval} --legends ExpA --criterion min-hter

  [Min. criterion: MIN-HTER ] Threshold on Development set `ExpA`: -1.399200e-01
  =====================  ==============  ==============
  ..                     Development     Evaluation
  =====================  ==============  ==============
  Failure to Acquire     0.0%            0.0%
  False Match Rate       2.4% (24/1000)  1.9% (19/1000)
  False Non Match Rate   0.0% (0/250)    2.0% (5/250)
  False Accept Rate      2.4%            1.9%
  False Reject Rate      0.0%            2.0%
  Half Total Error Rate  1.2%            1.9%
  =====================  ==============  ==============



.. note::

  When evaluation scores are provided, the ``-e`` (``--eval``) option must be passed.
  See ``metrics --help`` for further options.

  The ``scores-{dev,eval}`` brace expansion expands the path to every element
  in the list. here: ``scores-dev scores-eval``.


.. _bob.bio.base.bob_bio_plots:

Plots
-----

Customizable plotting sub-commands are available in the :py:mod:`bob.bio.base`
module (command: ``$ bob bio``). They take a list of development and/or
evaluation files and generate a single PDF file containing the plots.

Available plots are:

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

  $ bob bio cmc -e -v --output 'my_cmc.pdf' dev-1.txt eval-1.txt dev-2.txt eval-2.txt

where ``my_cmc.pdf`` will contain CMC curves for the two experiments represented
by their respective *dev* and *eval* scores-file.

.. note::

  By default, ``det``, ``roc``, ``cmc``, and ``dir`` plot development and
  evaluation curves on different plots. You can forcefully gather everything in
  the same plot using the ``-ns`` (``--no-split``) option.

.. note::

  The ``--figsize`` and ``--style`` options are two powerful options that can
  dramatically change the appearance of your figures. Try them! (e.g.
  ``--figsize 12,10 --style grayscale``)

Bob bio evaluate
----------------

A convenient command ``evaluate`` is provided to generate multiple metrics and
plots for a list of experiments. It generates two ``metrics`` outputs with EER,
HTER, minDCF criteria, along with ``roc``, ``det``, ``epc``, and ``hist`` plots
for each experiment. For example::

.. code-block:: sh

  $ bob bio evaluate -e -v -l 'my_metrics.txt' -o 'my_plots.pdf' {sys1,sys2}/{dev,eval}

will output metrics and plots for the two experiments (dev and eval pairs) in
`my_metrics.txt` and `my_plots.pdf`, respectively.



.. include:: links.rst
