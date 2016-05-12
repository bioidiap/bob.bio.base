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
   bob.bio.base.database.Database
   bob.bio.base.database.DatabaseZT
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
   bob.bio.base.database.DatabaseBob
   bob.bio.base.database.DatabaseBobZT
   bob.bio.base.database.DatabaseFileList


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

   .. exec::
      import json
      from bob.bio.base.grid import PREDEFINED_QUEUES
      json_obj = json.dumps(PREDEFINED_QUEUES, sort_keys=True, indent=2)
      json_obj = json_obj.replace("\n", "\n   ")
      print ('.. code-block:: JavaScript\n\n   PREDEFINED_QUEUES = %s\n\n' % json_obj)


.. include:: links.rst
