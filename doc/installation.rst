.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.installation:

=========================
Installation Instructions
=========================

As noted before, this package is part of the ``bob.bio`` packages, which in turn are part of the signal-processing and machine learning toolbox Bob_.
To install `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_, please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`_.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

.. note::
  Currently, running Bob_ under MS Windows in not yet supported.
  However, we found that running Bob_ in a virtual Unix environment such as the one provided by VirtualBox_ is a good alternative.

The most simple and most convenient way to use the ``bob.bio`` tools is to use a ``zc.buildout`` package, as explained in more detail `here <https://github.com/idiap/bob/wiki/Installation#using-zcbuildout-for-production>`__.
There, in the ``eggs`` section of the ``buildout.cfg`` file, simply list the ``bob.bio`` packages that you want, like:

.. code-block:: python

   eggs = bob.bio.base
          bob.bio.face
          bob.bio.gmm
          bob.bio.video
          bob.db.youtube
          gridtk

in order to download and install all packages that are required for your experiments.
In the example above, you might want to run a video face recognition experiments using the :py:class:`bob.bio.face.preprocessor.FaceDetector` and the :py:class:`bob.bio.face.extractor.DCTBlocks` feature extractor defined in :ref:`bob.bio.face <bob.bio.face>`, the :py:class:`bob.bio.gmm.algorithm.IVector` algorithm defined in :ref:`bob.bio.gmm <bob.bio.gmm>` and the video extensions defined in :ref:`bob.bio.video <bob.bio.video>`, using the YouTube faces database interface defined in :ref:`bob.db.youtube <bob.db.youtube>`.
Running the simple command line:

.. code-block:: sh

   $ python bootstrap-buildout.py
   $ ./bin/buildout

will the download and install all dependent packages locally (relative to your current working directory), and create a ``./bin`` directory containing all the necessary scripts to run the experiments.


Databases
~~~~~~~~~

With ``bob.bio`` you will run biometric recognition experiments using some default biometric recognition databases.
Though the verification protocols are implemented in ``bob.bio``, the original data are **not included**.
To download the original data of the databases, please refer to the according Web-pages.
For a list of supported databases including their download URLs, please refer to the :ref:`verification_databases`.

After downloading the original data for the databases, you will need to tell ``bob.bio``, where these databases can be found.
For this purpose, we have decided to implement a special file, where you can set your directories.
By default, this file is located in ``~/.bob_bio_databases.txt``, and it contains several lines, each line looking somewhat like:

.. code-block:: text

   [YOUR_ATNT_DIRECTORY] = /path/to/your/directory

.. note::
   If this file does not exist, feel free to create and populate it yourself.


Please use ``./bin/databases.py`` for a list of known databases, where you can see the raw ``[YOUR_DATABASE_PATH]`` entries for all databases that you haven't updated, and the corrected paths for those you have.


.. note::
   If you have installed only ``bob.bio.base``, there is no database listed -- as all databases are included in other packages, such as :ref:`bob.bio.face <bob.bio.face>` or :ref:`bob.bio.spear <bob.bio.spear>`.


Test your Installation
~~~~~~~~~~~~~~~~~~~~~~

One of the scripts that were generated during the bootstrap/buildout step is a test script.
To verify your installation, you should run the script running the nose tests for each of the ``bob.bio`` packages:

.. code-block:: sh

  $ ./bin/nosetests -vs bob.bio.base
  $ ./bin/nosetests -vs bob.bio.gmm
  ...

Some of the tests that are run require the images of the `AT&T database`_ database.
If the database is not found on your system, it will automatically download and extract the `AT&T database`_ a temporary directory, **which will not be erased**.

To avoid the download to happen each time you call the nose tests, please:

1. Download the `AT&T database`_ database and extract it to the directory of your choice.
2. Set an environment variable ``ATNT_DATABASE_DIRECTORY`` to the directory, where you extracted the database to.
   For example, in a ``bash`` you can call:

.. code-block:: sh

  $ export ATNT_DATABASE_DIRECTORY=/path/to/your/copy/of/atnt

.. note::
  To set the directory permanently, you can also change the ``atnt_default_directory`` in the file `bob/bio/base/test/utils.py <file:../bob/bio/base/test/utils.py>`_.
  In this case, there is no need to set the environment variable any more.

In case any of the tests fail for unexplainable reasons, please file a bug report through the `GitHub bug reporting system`_.

.. note::
  Usually, all tests should pass with the latest stable versions of the Bob_ packages.
  In other versions, some of the tests may fail.


Generate this documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generally, the documentation of this package is `available online <http://pythonhosted.org/bob.bio.base>`__, and this should be your preferred resource.
However, to generate this documentation locally, you call:

.. code-block:: sh

  $ ./bin/sphinx-build doc sphinx

Afterward, the documentation is available and you can read it, e.g., by using:

.. code-block:: sh

  $ firefox sphinx/index.html


.. _buildout.cfg: file:../buildout.cfg

.. include:: links.rst
