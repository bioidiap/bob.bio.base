.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

==============================
More about Running Experiments
==============================

Now that we have learned the implementation details, we can have a closer look into how experiments can be parametrized.

.. _running_part_2:

Running Experiments (part II)
-----------------------------

As mentioned before, running biometric recognition experiments can be achieved using the ``./bin/verify.py`` command line.
In section :ref:`running_part_1`, we have used registered resources to run an experiment.
However, the command line options of ``./bin/verify.py`` is more flexible, as you can have three different ways of defining tools:

1. Choose a resource (see ``./bin/resources.py`` or ``./bin/verify.py --help`` for a list of registered resources):

   .. code-block:: sh

      $ ./bin/verify.py --algorithm pca


2. Use a configuration file. Make sure that your configuration file has the correct variable name:

   .. code-block:: sh

      $ ./bin/verify.py --algorithm bob/bio/base/config/algorithm/pca.py


3. Instantiate a class on the command line. Usually, quotes ``"..."`` are required, and the ``--imports`` need to be specified:

   .. code-block:: sh

      $ ./bin/verify.py --algorithm "bob.bio.base.algorithm.PCA(subspace_dimension = 30, distance_function = scipy.spatial.distance.euclidean, is_distance_function = True)" --imports bob.bio.base scipy.spatial

All these three ways can be used for any of the five command line options: ``--database``, ``--preprocessor``, ``--extractor``, ``--algorithm`` and ``--grid``.
You can even mix these three types freely in a single command line.


Score Level Fusion of Different Algorithms on the same Database
---------------------------------------------------------------

In several of our publications, we have shown that the combination of several biometric recognition algorithms is able to outperform each single algorithm.
This is particularly true, when the algorithms rely on different kind of data, e.g., we have `fused face and speaker recognition system on the MOBIO database <http://publications.idiap.ch/index.php/publications/show/2688>`__.
As long as several algorithms are executed on the same database, we can simply generate a fusion system by using the ``./bin/fuse_scores.py`` script, generating a new score file:

.. code-block:: sh

   $ ./bin/fuse_scores.py --dev

This computation is based on the :py:class:`bob.learn.linear.CGLogRegTrainer`, which is trained on the scores of the development set files (``--dev-files``) for the given systems.
Afterwards, the fusion is applied to the ``--dev-files`` and the resulting score file is written to the file specified by ``--fused-dev-file``.
If ``--eval-files`` are specified, the same fusion that is trained on the development set is now applied to the evaluation set as well, and the ``--fused-eval-file`` is written.

.. note::
   When ``--eval-files`` are specified, they need to be in the same order as the ``dev-files``, otherwise the result is undefined.

The resulting ``--fused-dev-file`` and ``fused-eval-file`` can then be evaluated normally, e.g., using the ``./bin/evaluate.py`` script.


.. _grid-search:

Finding the Optimal Configuration
---------------------------------

Sometimes, configurations of tools (preprocessors, extractors or algorithms) are highly dependent on the database or even the employed protocol.
Additionally, configuration parameters depend on each other.
``bob.bio`` provides a relatively simple set up that allows to test different configurations in the same task, and find out the best set of configurations.
For this, the ``./bin/grid_search.py`` script can be employed.
This script executes a configurable series of experiments, which reuse data as far as possible.
Please check out ``./bin/grid_search.py --help`` for a list of command line options.

The Configuration File
~~~~~~~~~~~~~~~~~~~~~~
The most important parameter to the ``./bin/grid_search.py`` is the ``--configuration-file``.
In this configuration file it is specified, which parameters of which part of the algorithms will be tested.
An example for a configuration file can be found in the test scripts: ``bob/bio/base/test/dummy/grid_search.py``.
The configuration file is a common python file, which can contain certain variables:

1. ``preprocessor =``
2. ``extractor =``
3. ``algorithm =``
4. ``replace =``
5. ``requirement =``
6. ``imports =``

The variables from 1. to 3. usually contain instantiations for classes of :ref:`bob.bio.base.preprocessors`, :ref:`bob.bio.base.extractors` and :ref:`bob.bio.base.algorithms`, but also registered :ref:`bob.bio.base.resources` can be used.
For any of the parameters of the classes, a *placeholder* can be put.
By default, these place holders start with a # character, followed by a digit or character.
The variables 1. to 3. can also be overridden by the command line options ``--preprocessor``, ``--extractor`` and ``--algorithm`` of the ``./bin/grid_search.py`` script.

The ``replace`` variable has to be set as a dictionary.
In it, you can define with which values your place holder key should be filled, and in which step of the tool chain execution this should happen.
The steps are ``'preprocess'``, ``'extract'``, ``'project'``, ``'enroll'`` and ``'score'``.
For each of the steps, it can be defined, which placeholder should be replaced by which values.
To be able to differentiate the results later on, each of the replacement values is bound to a directory name.
The final structure looks somewhat like that:

.. code-block:: python

  replace = {
      step1 : {
          '#a' : {
              'Dir_a1' : 'Value_a1',
              'Dir_a2' : 'Value_a2'
           },

          '#b' : {
              'Dir_b1' : 'Value_b1',
              'Dir_b2' : 'Value_b2'
          }
      },

      step2 : {
          '#c' : {
              'Dir_c1' : 'Value_c1',
              'Dir_c2' : 'Value_c2'
          }
      }
  }


Of course, more than two values can be selected.
In the above example, the results of the experiments will be placed into a directory structure as ``results/[...]/Dir_a1/Dir_b1/Dir_c1/[...]``.

.. note::
   Please note that we are using a dictionary structure to define the replacements.
   Hence, the order of the directories inside the same step might not be in the same order as written in the configuration file.
   For the above example, a directory structure of `results/[...]/Dir_b1/Dir_a1/Dir_c1/[...]`` might be possible as well.


Additionally, tuples of place holders can be defined, in which case always the full tuple will be replaced in one shot.
Continuing the above example, it is possible to add:

.. code-block:: python

  ...
      step3 : {
          '(#d,#e)' : {
              'Dir_de1' : ('Value_d1', 'Value_e1'),
              'Dir_de2' : ('Value_d2', 'Value_e2')
          }
      }

.. warning::
   *All possible combinations* of the configuration parameters are tested, which might result in a *huge number of executed experiments*.

Some combinations of parameters might not make any sense.
In this case, a set of requirements on the parameters can be set, using the ``requirement`` variable.
In the requirements, any string including any placeholder can be put that can be evaluated using pythons ``eval`` function:

.. code-block:: python

  requirement = ['#a > #b', '2*#c != #a', ...]

Finally, when any of the classes or variables need to import a certain python module, it needs to be declared in the ``imports`` variable.
If you, e.g., test, which ``scipy.spatial`` distance function works best for your features, please add the imports (and don't forget the ``bob.bio.base`` and other ``bob.bio`` packages in case you use their tools):

.. code-block:: python

  imports = ['scipy', 'bob.bio.base', 'bob.bio.face']


Further Command Line Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``./bin/grid_search.py`` script has a further set of command line options.

- The ``--database`` and the ``--protocol`` define, which database and (optionally) which protocol should be used.
- The ``--sub-directory`` is similar to the one in the ``./bin/verify.py``.
- ``--result-directory`` and ``--temp-directory`` specify directories to write results and temporary files into. Defaults are ``./results/grid_search`` and ``./temp/grid_search`` in the current directory. Make sure that the ``--temp-directory`` can store sufficient amount of data.
- The ``--preprocessor``, ``--extractor`` and ``--algorithm`` can be used to override the ``preprocessor``, ``extractor`` and ``algorithm`` fields in the configuration file (in which case the configuration file does not need to contain these variables).
- The ``--grid`` option can select the SGE_ configuration.
- The ``--parallel`` option can run on the local machine using the given number of parallel threads.
- The ``--preprocessed-directory`` can be used to select a directory of previously preprocessed data. This should not be used in combination with testing different preprocessor parameters.
- The ``--gridtk-database-directory`` can be used to select another directory, where the ``submitted.sql3`` files will be stored.
- Sometimes, the gridtk databases grow, and are too large for holding all experiments. Using the ``--gridtk-database-split-level``, databases can be split at the desired level.
- The ``--write-commands`` directory can be selected to write the executed commands into (this is useful in case some experiments fail and need to be rerun).
- The ``--dry-run`` flag should always be used before the final execution to see if the experiment definition works as expected.
- The ``--skip-when-existent`` flag will only execute the experiments that have not yet finished (i.e., where the resulting score files are not produced yet).
- With the ``--executable`` flag, you might select a different script rather that ``bob.bio.base.script.verify`` to run the experiments (such as the ``bob.bio.gmm.script.verify_gmm``).
- Finally, additional options might be sent to the ``./bin/verify.py`` script directly. These options might be put after a ``--`` separation.


Evaluation of Results
~~~~~~~~~~~~~~~~~~~~~

To evaluate a series of experiments, a special script iterates through all the results and computes EER on the development set and HTER on the evaluation set, for both the ``nonorm`` and the ``ztnorm`` directories.
Simply call:

.. code-block:: sh

  $ ./bin/collect_results.py -vv --directory [result-base-directory] --sort

This will iterate through all result files found in ``[result-base-directory]`` and sort the results according to the EER on the development set (the sorting criterion can be modified using the ``--criterion``  and the ``--sort-key`` comamnd line options).
Hence, to find the best results of your grid search experiments (with default directories), simply run:

.. code-block:: sh

  $ ./bin/collect_results.py -vv --directory results/grid_search --sort --criterion EER --sort-key nonorm-dev




.. include:: links.rst
