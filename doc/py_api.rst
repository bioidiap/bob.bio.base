
===========================
Python API for bob.bio.base
===========================

Generic functions
-----------------

IO-related functions
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.read_original_data
   bob.bio.base.load
   bob.bio.base.save
   bob.bio.base.load_compressed
   bob.bio.base.save_compressed
   bob.bio.base.open_compressed
   bob.bio.base.close_compressed
   bob.bio.base.check_file
   bob.bio.base.vstack_features


Functions dealing with resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.load_resource
   bob.bio.base.read_config_file
   bob.bio.base.resource_keys
   bob.bio.base.extensions
   bob.bio.base.valid_keywords


Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.get_config
   bob.bio.base.score_fusion_strategy
   bob.bio.base.selected_elements
   bob.bio.base.selected_indices


Tools to run recognition experiments
------------------------------------

Command line generation
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.tools.command_line_parser
   bob.bio.base.tools.initialize
   bob.bio.base.tools.command_line
   bob.bio.base.tools.write_info
   bob.bio.base.tools.FileSelector

Controlling of elements
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.tools.groups
   bob.bio.base.tools.indices

Preprocessing
~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.tools.preprocess
   bob.bio.base.tools.read_preprocessed_data

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.tools.train_extractor
   bob.bio.base.tools.extract
   bob.bio.base.tools.read_features

Algorithm
~~~~~~~~~

.. autosummary::
   bob.bio.base.tools.train_projector
   bob.bio.base.tools.project
   bob.bio.base.tools.train_enroller
   bob.bio.base.tools.enroll

Scoring
~~~~~~~

.. autosummary::
   bob.bio.base.tools.compute_scores
   bob.bio.base.tools.concatenate
   bob.bio.base.tools.calibrate
   bob.bio.base.script.figure.Metrics

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
   bob.bio.base.script.figure.Dir
   bob.bio.base.script.figure.Hist

OpenBR conversions
------------------
.. autosummary::
   bob.bio.base.score.openbr.write_matrix
   bob.bio.base.score.openbr.write_score_file

Details
-------

.. automodule:: bob.bio.base

   .. attribute:: valid_keywords

      Valid keywords, for which resources are defined, are ``('database', 'preprocessor', 'extractor', 'algorithm', 'grid')``


.. automodule:: bob.bio.base.tools

   .. autoclass:: FileSelector

.. automodule:: bob.bio.base.score.load
.. automodule:: bob.bio.base.score.openbr
.. automodule:: bob.bio.base.script.figure
.. automodule:: bob.bio.base.script.commands
.. automodule:: bob.bio.base.script.gen

.. include:: links.rst
