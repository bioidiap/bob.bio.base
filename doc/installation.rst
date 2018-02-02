.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.installation:

Installation Instructions
=========================

As noted before, this package is part of the ``bob.bio`` packages, which in
turn are part of the signal-processing and machine learning toolbox Bob_. To
install Bob_, please read the `Installation Instructions <bobinstall_>`_.

Then, to install the ``bob.bio`` packages and in turn maybe the database
packages that you want to use, use conda_ to install them:

.. code-block:: sh

    $ conda search bob.bio  # searching
    $ conda search bob.db  # searching
    $ conda install bob.bio.base bob.bio.<bioname> bob.db.<dbname>

where you would replace ``<bioname>`` and ``<dbname>`` with the name of
packages that you want to use.

An example installation
-----------------------

For example, you might want to run a video face recognition experiments using
the :py:class:`bob.bio.face.preprocessor.FaceDetect` and the
:py:class:`bob.bio.face.extractor.DCTBlocks` feature extractor defined in
:ref:`bob.bio.face <bob.bio.face>`, the
:py:class:`bob.bio.gmm.algorithm.IVector` algorithm defined in
:ref:`bob.bio.gmm <bob.bio.gmm>` and the video extensions defined in
:ref:`bob.bio.video <bob.bio.video>`, using the YouTube faces database
interface defined in :ref:`bob.db.youtube <bob.db.youtube>`. Running the
command line below will install all the required packages:

.. code-block:: sh

    $ source activate <bob_conda_environment>
    $ conda install bob.bio.base \
                    bob.bio.face \
                    bob.bio.gmm \
                    bob.bio.video \
                    bob.db.youtube \
                    gridtk


Databases
---------

With ``bob.bio`` you will run biometric recognition experiments using biometric
recognition databases. Though the verification protocols are implemented in
``bob.bio``, the raw data are **not included**. To download the raw
data of the databases, please refer to the according Web-pages. For a list of
supported databases including their download URLs, please refer to the
`biometric recognition databases`_.

After downloading the raw data for the databases, you will need to tell
``bob.bio``, where these databases can be found. For this purpose, we have
decided to implement a special file, where you can set your directories. By
default, this file is located in ``~/.bob_bio_databases.txt``, and it contains
several lines, each line looking somewhat like:

.. code-block:: text

   [YOUR_ATNT_DIRECTORY] = /path/to/your/directory

.. note::

   If this file does not exist, feel free to create and populate it yourself.


Please use ``databases.py`` for a list of known databases, where you can see
the raw ``[YOUR_DATABASE_PATH]`` entries for all databases that you haven't
updated, and the corrected paths for those you have.

.. note::

   If you have installed only ``bob.bio.base``, there is no database listed --
   as all databases are included in other packages, such as
   :ref:`bob.bio.face <bob.bio.face>` or :ref:`bob.bio.spear <bob.bio.spear>`.
   Also, please don't forget that you need to install the corresponding
   ``bob.db.<name>`` package as well.


Test your Installation
----------------------

You can install the ``nose`` package to test your installation and use that to
verify your installation:

.. code-block:: sh

  $ conda install nose  # install nose
  $ nosetests -vs bob.bio.base
  $ nosetests -vs bob.bio.gmm
  ...

You should run the script running the nose tests for each of the ``bob.bio``
packages separately.

.. code-block:: sh

  $ nosetests -vs bob.bio.base
  $ nosetests -vs bob.bio.gmm
  ...

Some of the tests that are run require the images of the `AT&T database`_
database. If the database is not found on your system, it will automatically
download and extract the `AT&T database`_ a temporary directory, **which will
not be erased**.

To avoid the download to happen each time you call the nose tests, please:

1. Download the `AT&T database`_ database and extract it to the directory of
   your choice.
2. Set an environment variable ``ATNT_DATABASE_DIRECTORY`` to the directory,
   where you extracted the database to. For example, in a ``bash`` you can
   call:

.. code-block:: sh

  $ export ATNT_DATABASE_DIRECTORY=/path/to/your/copy/of/atnt


In case any of the tests fail for unexplainable reasons, please send a report
through our `mailing list`_.

.. note::
  Usually, all tests should pass with the latest stable versions of Bob_
  packages. In other versions, some of the tests may fail.


.. include:: links.rst
