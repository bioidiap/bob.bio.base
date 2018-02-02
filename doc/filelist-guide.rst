.. vim: set fileencoding=utf-8 :
.. @author: Manuel Guenther <manuel.guenther@idiap.ch>
.. @date:   Fri Aug 29 13:52:39 CEST 2014

Verification File List Database Guide
=====================================

The Database Interface
----------------------

The :py:class:`bob.bio.base.database.FileListBioDatabase` complies with the standard biometric verification database as described in :ref:`bob.bio.base`.
All functions defined in that interface are properly instantiated, as soon as the user provides the required file lists.

Creating File Lists
-------------------

The initial step for using this package is to provide file lists specifying the ``'world'`` (training; optional), ``'dev'`` (development; required) and ``'eval'`` (evaluation; optional) set to be used by the biometric verification algorithm.
The summarized complete structure of the list base directory (here denoted as ``basedir``) containing all the files should be like this::

    filelists_directory
         |-- norm
               |-- train_world.lst
               |-- train_optional_world_1.lst
               |-- train_optional_world_2.lst
         |
         |-- dev
               |-- for_models.lst
               |-- for_probes.lst
               |-- for_scores.lst
               |-- for_tnorm.lst
               |-- for_znorm.lst
         |
         |-- eval
               |-- for_models.lst
               |-- for_probes.lst
               |-- for_scores.lst
               |-- for_tnorm.lst
               |-- for_znorm.lst


The file lists contain several information that need to be available for the biometric recognition experiment to run properly.
A complete list of possible information is:

* ``filename``: The name of the data file, **relative** to the common root of all data files, and **without** file name extension.
* ``client_id``: The name or ID of the subject the biometric traces of which are contained in the data file.
  These names are handled as :py:class:`str` objects, so ``001`` is different from ``1``.
* ``model_id``:

  - used for model enrollment: The name or ID of the *client model* that should be enrolled. In most cases, the ``model_id`` is identical to the ``client_id``.
  - used for scoring: The name or ID of the *client model* that the probe file should be compared with.

* ``claimed_client_id``:

  - used for scoring: The ``client_id`` of the client model that the probe file should be compared with.


The following list files need to be created:

- **For training** (optional):

  * *world file*, with default name ``train_world.lst``, in the default sub-directory ``norm``.
    It is a 2-column file with format:

    .. code-block:: text

      filename client_id

  * two *world files*, with default names ``train_optional_world_1.lst`` and ``train_optional_world_2.lst``, in default sub-directory ``norm``.
    The format is the same as for the world file.
    These files are not needed for most of biometric recognition algorithms, hence, they need to be specified only if the algorithm uses them.

- **For enrollment**:

  * one or two *model files* for the development (and evaluation) set, with default name ``for_models.lst`` in the default sub-directories ``dev`` (and ``eval``).
    They are 3-column files with format:

    .. code-block:: text

      filename model_id client_id

- **For scoring**:

  There exist two different ways to implement file lists used for scoring.

  * The first (and simpler) variant is to define a file list of probe files, where all probe files will be tested against all models.
    Hence, you need to specify one or two *probe files* for the development (and evaluation) set, with default name ``for_probes.lst`` in the  default sub-directories ``dev`` (and ``eval``).
    They are 2-column files with format:

    .. code-block:: text

      filename client_id

  * The other option is to specify a detailed list, which probe file should be be compared with which client model, i.e., one or two *score files* for the development (and evaluation) set, with default name ``for_scores.lst`` in the  sub-directories ``dev`` (and ``eval``).
    These files need to be provided only if the scoring is to be done selectively, meaning by creating a sparse probe/model scoring matrix.
    They are 4-column files with format:

    .. code-block:: text

      filename model_id claimed_client_id client_id

  .. note:: The verification queries will use either only the probe or only the score files, so only one of them is mandatory.
            If only one of the two files is available, the scoring technique will be automatically determined.
            In case both probe and score files are provided, the user should set the parameter ``use_dense_probe_file_list``, which specifies the files to consider, when creating the object of the ``Database`` class.


- **For ZT score normalization** (optional):

  Optionally, file lists for ZT score normalization can be added.
  These are:

  * one or two *files for t-score normalization* for the development (and evaluation) set, with default name ``for_tnorm.lst`` in both sub-directories ``dev`` (and ``eval``).
    They are 3-column files with format:

    .. code-block:: text

      filename model_id client_id

  * one or two *files for z-score normalization* for the development (and evaluation) set, with default name ``for_znorm.lst`` in both sub-directories ``dev`` (and ``eval``).
    They are 2-column files with format:

    .. code-block:: text

      filename client_id


Please note that in all files, the lines starting with any number of white
space and ``#`` will be ignored.


Protocols and File Lists
------------------------

When you instantiate a database, you have to specify the base directory that contains the file lists.
If you have only a single protocol, you could specify the full path to the file lists described above as follows:

.. code-block:: python

  >>> db = bob.bio.base.database.FileListBioDatabase('basedir/protocol', 'mydb')

Next, you should query the data, WITHOUT specifying any protocol:

.. code-block:: python

  >>> db.objects()

Alternatively, if you have more protocols, you could do the following:

.. code-block:: python

  >>> db = bob.bio.base.database.FileListBioDatabase('basedir', 'mydb', protocol='protocol')
  >>> db.objects()

or specify the protocol while querying the database:

.. code-block:: python

  >>> db = bob.bio.base.database.FileListBioDatabase('basedir', 'mydb')
  >>> db.objects(protocol='protocol')

When a protocol is specified, it is appended to the base directory that contains the file lists.
You can query the database with ``another`` protocol, simply as:

.. code-block:: python

  >>> db = bob.bio.base.database.FileListBioDatabase('basedir', 'mydb')
  >>> db.objects(protocol='protocol')
  >>> db.objects(protocol='another')

and you retrieve the files stored in `basedir/protocol` and `basedir/another`, respectively.

.. note::
   If you use several protocols as explained above, the ``use_dense_probe_file_list`` parameter is global for all protocols.
   In case you have ``for_scores.lst`` in one and ``for_probes.lst`` in another protocol, it will automatically switch between the scoring strategies -- as long as you leave ``use_dense_probe_file_list=None``.
