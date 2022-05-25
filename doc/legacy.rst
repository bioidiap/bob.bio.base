.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: Wed 18 Aug 2020 10:21:00 UTC+02

.. _bob.bio.base.legacy:

=================================================
 Connecting legacy bob.bio.base to PipelineSimple
=================================================

The transition to the pipeline concept changed the way data goes from the raw sample to the extracted features, and how the biometric algorithm is applied.
However, a set of tools was implemented to support the older bob implementations (designated as *legacy*) of database, preprocessor, extractor, and algorithms.


This adaptation consists of wrapper classes that take a legacy bob class as input and constructs a Transformer or :any:`bob.bio.base.pipelines.BioAlgorithm` out of it.


.. WARNING::

  A temporary folder is created in case a legacy bob package needs to write on disk during its operation.
  However, this folder is persistent between experiments. You should remove its content before running another experiment.


Legacy FileList Database interface
----------------------------------

This is a similar database interface to :ref:`the CSV file interface <bob.bio.base.database.csv_file_interface>`, but takes information from a series of two- or three-column files without header instead of CSV files and returns a legacy database (use a ``Database Connector (was removed) <bob.bio.base.legacy.database_connector>`` to create a database interface).


The files are separated into three sets: ``'world'`` (training; optional), ``'dev'`` (development; required) and ``'eval'`` (evaluation; optional) set to be used by the biometric verification algorithm.
The summarized complete structure of the list base directory (here denoted as ``basedir``) containing all the files should be like this:

.. code-block:: text

  filelists_directory
  |
  +-- norm
  |   |
  |   +-- train_world.lst
  |   +-- train_optional_world_1.lst
  |   +-- train_optional_world_2.lst
  |
  +-- dev
  |   |
  |   +-- for_models.lst
  |   +-- for_probes.lst
  |   +-- for_scores.lst
  |   +-- for_tnorm.lst
  |   +-- for_znorm.lst
  |
  +-- eval
      |
      +-- for_models.lst
      +-- for_probes.lst
      +-- for_scores.lst
      +-- for_tnorm.lst
      +-- for_znorm.lst


The file lists contain several information that need to be available for the biometric recognition experiment to run properly.
A complete list of possible information is:

* ``filename``: The name of the data file, **relative** to the common root of all data files, and **without** file name extension.
* ``client_id``: The name or ID of the subject the biometric traces of which are contained in the data file.
  These names are handled as :py:class:`str` objects, so ``001`` is different from ``1``.
* ``model_id``:

  - used for model enrollment: The name or ID of the *client model* that should be enrolled. In most cases, the ``model_id`` is identical to the ``client_id``.
  - used for scoring: The name or ID of the *client model* that the probe file should be compared with.

* ``claimed_client_id``:

  - used for scoring: The ``client_id`` of the client model that the probe file should be compared with.


The following list files need to be created:

- **For training** (optional):

  * *world file*, with default name ``train_world.lst``, in the default sub-directory ``norm``.
    It is a 2-column file with format:

    .. code-block:: text

      filename client_id

  * two *world files*, with default names ``train_optional_world_1.lst`` and ``train_optional_world_2.lst``, in default sub-directory ``norm``.
    The format is the same as for the world file.
    These files are not needed for most biometric recognition algorithms, hence, they need to be specified only if the algorithm uses them.

- **For enrollment**:

  * one or two *model files* for the development (and evaluation) set, with default name ``for_models.lst`` in the default sub-directories ``dev`` (and ``eval``).
    They are 3-column files with format:

    .. code-block:: text

      filename model_id client_id

- **For scoring**:

  There exist two different ways to implement file lists used for scoring.

  * The first (and simpler) variant is to define a file list of probe files, where all probe files will be tested against all models.
    Hence, you need to specify one (or two) *probe files* for the development (and evaluation) set, with the default name ``for_probes.lst`` in the default sub-directory ``dev`` (and ``eval``).
    They are 2-column files with format:

    .. code-block:: text

      filename client_id

  * The other option is to specify a detailed list of which probe files should be compared with which client model, i.e., one (or two) *score files* for the development (and evaluation) set, with the default name ``for_scores.lst`` in the sub-directory ``dev`` (and ``eval``).
    These files need to be provided only if the scoring is to be done selectively, meaning by creating a sparse probe/model scoring matrix.
    They are 4-column files with format:

    .. code-block:: text

      filename model_id claimed_client_id client_id

  .. note:: The verification queries will use either only the probe or only the score files, so only one of them is mandatory.
            If only one of the two files is available, the scoring technique will be automatically determined.
            In case both probe and score files are provided, the user should set the parameter ``use_dense_probe_file_list``, which specifies the files to consider, when creating the object of the ``Database`` class.


- **For ZT score normalization** (optional):

  Optionally, file lists for ZT score normalization can be added.
  These are:

  * one or two *files for t-score normalization* for the development (and evaluation) set, with default name ``for_tnorm.lst`` in both sub-directories ``dev`` (and ``eval``).
    They are 3-column files with format:

    .. code-block:: text

      filename model_id client_id

  * one or two *files for z-score normalization* for the development (and evaluation) set, with default name ``for_znorm.lst`` in both sub-directories ``dev`` (and ``eval``).
    They are 2-column files with format:

    .. code-block:: text

      filename client_id

.. note::

  In all files, the lines starting with any number of white space and ``#``
  will be ignored.


Legacy Preprocessor wrapper
---------------------------

The :py:class:`bob.bio.base.transformers.PreprocessorTransformer` wrapper takes
a :py:class`bob.bio.base.preprocessor` from the old :py:mod:`bob.bio.base` as
input and creates a Transformer out of it. The
``bob.bio.base.preprocessor.Preprocessor.__call__`` method of the
:py:class`~bob.bio.base.preprocessor.Preprocessor` class is called when the
``Transformer.transform`` method is called.


This example shows how to create a Transformer out of a legacy preprocessor (FaceCrop, from bob.bio.face):

.. code-block:: python

    from bob.bio.face.preprocessor import FaceCrop
    from bob.bio.base.transformers import PreprocessorTransformer

    # Initialize the legacy Preprocessor
    legacy_preprocessor = FaceCrop(
        cropped_size=(80,64),
        cropped_positions={'leye':'16,15', 'reye':'16,48'},
        fixed_positions={'leye':'50,24', 'reye','50,64'}
    )

    # Create the Transformer
    preprocessor_transformer = PreprocessorTransformer( legacy_preprocessor )


Legacy Extractor wrapper
------------------------

A similar wrapper is available for the legacy :py:mod:`bob.bio.base` Extractor. It is the :py:class:`bob.bio.base.transformers.ExtractorTransformer`.
It maps the ``Transformer.transform`` method to the ``bob.bio.base.extractor.Extractor.__call__`` of the legacy Extractor.

Here is an example showing how to create a Transformer from a legacy Extractor (Linearize, from bob.bio.base):

.. code-block:: python

    from bob.bio.base.extractor import Linearize
    from bob.bio.base.transformers import ExtractorTransformer

    # Create the Transformer from the legacy Extractor
    extractor_transformer = ExtractorTransformer( Linearize() )

.. include:: links.rst
