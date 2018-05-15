.. _bob.bio.base.baseline:

==================
Defining baselines
==================


Once you have a biometric system well established, tuned and working for a particular database (or a particular set of databases), you may want to provide **an easier to reproduce** way to share it.
For this purpose, we defined something called baseline.

A baseline is composed by the triplet :any:`bob.bio.base.preprocessor.Preprocessor`, :any:`bob.bio.base.extractor.Extractor` and :any:`bob.bio.base.algorithm.Algorithm`.

First, check it out the baselines ready to be triggered in your environment by doing:

.. code-block:: sh

    $ bob bio baseline --help


To create your own baseline, you just need to define it like in the recipe below:

.. code-block:: py

    from bob.bio.base.baseline import Baseline

    class DummyBaseline(Baseline):

        def __init__(self):

            self.preprocessors = dict() # SHOULD BE DEFINED AS A DICTIONARY
            self.preprocessors["default"] = 'my-preprocessor'
            self.extractor = 'my-extractor'
            self.algorithm = 'my-algorithm'
    baseline = DummyBaseline()

Some databases may require some especific preprocessors depending on the type of meta-informations provided.
For instance, for some face recognition databases, faces should be cropped in a particular way depending on the annotations provided. 
To approach this issue, the preprocessors are defined in a dictionary, with a generic preprocessor defined as **default** and the database specific preprocessor defined by database name as in the example below:

.. code-block:: py

    self.preprocessors = dict()
    self.preprocessors["default"] = 'my-preprocessor'
    self.preprocessors["database_name"] = 'my-specific-preprocessor'
 

Follow below a full example on how to define a baseline with database specific preprocessors.

.. code-block:: py

    from bob.bio.base.baseline import Baseline

    class AnotherBaseline(Baseline):

        def __init__(self):

            self.preprocessors = dict() # SHOULD BE DEFINED AS A DICTIONARY
            self.preprocessors["default"] = 'my-preprocessor'
            self.preprocessors["database_name"] = 'my-specific-preprocessor'
            self.extractor = 'my-extractor'
            self.algorithm = 'my-algorithm'
    baseline = AnotherBaseline()

.. note::

   The triplet can be a resource or a configuration file.
   This works in the same way as in :ref:`Running Experiments <running_part_1>`.

.. note::

  Baselines are also registered as resources under the keyworkd `bob.bio.baseline`. 

You can find the list of readily available baselines using the ``resources.py`` command:

.. code-block:: sh

    $ resources.py --types baseline
