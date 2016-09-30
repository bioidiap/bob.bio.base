.. _bob.bio.base.implemented:

=================================
Tools implemented in bob.bio.base
=================================

Summary
-------

Base Classes
~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.preprocessor.Preprocessor
   bob.bio.base.extractor.Extractor
   bob.bio.base.algorithm.Algorithm
   bob.bio.base.grid.Grid


Implementations
~~~~~~~~~~~~~~~

.. autosummary::
   bob.bio.base.preprocessor.Filename
   bob.bio.base.extractor.Linearize
   bob.bio.base.algorithm.Distance
   bob.bio.base.algorithm.PCA
   bob.bio.base.algorithm.LDA
   bob.bio.base.algorithm.PLDA
   bob.bio.base.algorithm.BIC
   bob.bio.base.database.BioDatabase
   bob.bio.base.database.ZTBioDatabase


Preprocessors
-------------

.. automodule:: bob.bio.base.preprocessor

Extractors
----------

.. automodule:: bob.bio.base.extractor

Algorithms
----------

.. automodule:: bob.bio.base.algorithm

Databases
---------

.. automodule:: bob.bio.base.database

Grid Configuration
------------------

.. automodule:: bob.bio.base.grid

.. data:: PREDEFINED_QUEUES

   A dictionary of predefined queue keywords, which are adapted to the Idiap_ SGE.


   .. adapted from http://stackoverflow.com/a/29789910/3301902 to ge a nice dictionary content view


.. include:: links.rst
