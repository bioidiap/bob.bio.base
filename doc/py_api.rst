
===========================
Python API for bob.bio.base
===========================


Pipelines
---------

Database
~~~~~~~~

.. autosummary::

  bob.bio.base.pipelines.Database
  bob.bio.base.pipelines.Database.background_model_samples
  bob.bio.base.pipelines.Database.references
  bob.bio.base.pipelines.Database.probes

Database implementations
........................

.. autosummary::

 bob.bio.base.database.CSVDatabase

Biometric Algorithm
~~~~~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.pipelines.BioAlgorithm
  bob.bio.base.pipelines.BioAlgorithm.create_templates
  bob.bio.base.pipelines.BioAlgorithm.compare

Writing Scores
~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.pipelines.ScoreWriter
  bob.bio.base.pipelines.FourColumnsScoreWriter
  bob.bio.base.pipelines.CSVScoreWriter

Assembling the pipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.pipelines.PipelineSimple
  bob.bio.base.pipelines.PipelineScoreNorm


Creating Transformers from legacy constructs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.transformers.PreprocessorTransformer
  bob.bio.base.transformers.ExtractorTransformer


Legacy Constructs
-----------------

Base classes
~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.preprocessor.Preprocessor
  bob.bio.base.extractor.Extractor

Implementations
~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.algorithm.Distance
  bob.bio.base.algorithm.GMM
  bob.bio.base.algorithm.ISV
  bob.bio.base.algorithm.JFA
  bob.bio.base.transformers.ReferenceIdEncoder
  bob.bio.base.database.AtntBioDatabase


Generic functions
-----------------

Functions dealing with resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.utils.resources.load_resource
  bob.bio.base.utils.resources.read_config_file
  bob.bio.base.utils.resources.resource_keys
  bob.bio.base.utils.resources.extensions
  bob.bio.base.utils.resources.valid_keywords


Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.score_fusion_strategy
  bob.bio.base.selected_elements
  bob.bio.base.selected_indices
  bob.bio.base.utils.annotations.read_annotation_file



Loading data
------------

.. autosummary::

  bob.bio.base.score.load.open_file
  bob.bio.base.score.load.scores
  bob.bio.base.score.load.split
  bob.bio.base.score.load.cmc
  bob.bio.base.score.load.four_column
  bob.bio.base.score.load.split_four_column
  bob.bio.base.score.load.cmc_four_column
  bob.bio.base.score.load.five_column
  bob.bio.base.score.load.split_five_column
  bob.bio.base.score.load.cmc_five_column

Plotting
--------

.. autosummary::

  bob.bio.base.script.figure.Cmc
  bob.bio.base.script.figure.Det
  bob.bio.base.script.figure.Dir
  bob.bio.base.script.figure.Hist
  bob.bio.base.script.figure.Roc



IO-related functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::

  bob.bio.base.load
  bob.bio.base.save
  bob.bio.base.load_compressed
  bob.bio.base.save_compressed
  bob.bio.base.open_compressed
  bob.bio.base.close_compressed
  bob.bio.base.check_file


Details
-------

.. automodule:: bob.bio.base

.. automodule:: bob.bio.base.annotator
.. automodule:: bob.bio.base.pipelines
.. automodule:: bob.bio.base.database
.. automodule:: bob.bio.base.preprocessor
.. automodule:: bob.bio.base.extractor
.. automodule:: bob.bio.base.transformers
.. automodule:: bob.bio.base.algorithm
.. automodule:: bob.bio.base.score.load
.. automodule:: bob.bio.base.script.figure
.. automodule:: bob.bio.base.script.commands
.. automodule:: bob.bio.base.script.gen
.. automodule:: bob.bio.base.utils
.. automodule:: bob.bio.base.utils.resources
.. automodule:: bob.bio.base.utils.io
.. automodule:: bob.bio.base.utils.annotations





.. include:: links.rst
