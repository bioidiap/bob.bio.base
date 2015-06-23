


=================================
Tools implemented in bob.bio.base
=================================

Databases
---------

.. automodule:: bob.bio.base.database

Preprocessors
-------------

.. automodule:: bob.bio.base.preprocessor

Extractors
----------

.. automodule:: bob.bio.base.extractor

Algorithms
----------

.. automodule:: bob.bio.base.algorithm


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
