
===========================
Python API for bob.bio.base
===========================

Generic functions
-----------------

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
   bob.bio.base.vstack_features


Pipelines
~~~~~~~~~



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



Details
-------


.. automodule:: bob.bio.base.score.load
.. automodule:: bob.bio.base.script.figure
.. automodule:: bob.bio.base.script.commands
.. automodule:: bob.bio.base.script.gen

.. include:: links.rst
