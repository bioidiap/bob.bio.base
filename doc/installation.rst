.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.installation:

=========================
Installation Instructions
=========================

As noted before, this package is part of the ``bob.bio`` packages, which in
turn are part of the signal-processing and machine learning toolbox Bob_. To
install Bob_, please read the `Installation Instructions <bobinstall_>`_.

Then, to install the ``bob.bio`` packages and in turn maybe the database
packages that you want to use, use conda_ to install them:

.. code-block:: sh

    $ conda search "bob.bio.*"  # searching
    $ conda search "bob.db.*"  # searching
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
                    bob.db.youtube


Databases
---------

With ``bob.bio`` you will run biometric recognition experiments using biometric
recognition databases. Though the verification protocols are implemented in
``bob.bio``, the raw data are **not included**. To download the raw
data of the databases, please refer to the according Web-pages. For a list of
supported databases including their download URLs, please refer to the
`biometric recognition databases`_.

After downloading the **raw data** for the databases, you will need to tell
``bob.bio``, where these databases can be found. This can be set using
our :ref:`bob.extension.rc`.
For instance, the name convention to be followed in the `~/.bobrc` file is: `<database_package_name>.directory`. 
The command below shows how to set the path of the :ref:`bob.db.youtube`.

.. code-block:: sh

   bob config set bob.db.youtube.directory /path/to/the/youtube/database


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

In case any of the tests fail for unexplainable reasons, please send a report
through our `mailing list`_.

.. note::
  Usually, all tests should pass with the latest stable versions of Bob_
  packages. In other versions, some of the tests may fail.


.. include:: links.rst
