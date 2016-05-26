.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <Manuel.Guenther@idiap.ch>
.. Mon 23 04 2012

======================
Implementation Details
======================

The ``bob.bio`` module is specifically designed to be as flexible as possible while trying to keep things simple.
Therefore, it uses python to implement tools such as preprocessors, feature extractors and recognition algorithms.
It is file based so any tool can implement its own way of reading and writing data, features or models.
Configurations are stored in configuration files, so it should be easy to test different parameters of your algorithms without modifying the code.


Base Classes
------------

All tools implemented in the ``bob.bio`` packages are based on some classes, which are defined in the ``bob.bio.base`` package, and which are detailed below.
Most of the functionality is provided in the base classes, but any function can be overridden in the derived class implementations.

In the derived class constructors, the base class constructor needs to be called.
For automatically tracing the algorithms, all parameters that are passed to the derived class constructor should be passed to the base class constructor as a list of keyword arguments (which is indicated by ``...`` below).
This will assure that all parameters of the experiments are stored into the ``Experiment.info`` file.

.. note::
   All tools are based on reading, processing and writing files.
   By default, any type of file is allowed to be handled, and file names are provided to the ``read_...`` and ``write_...`` functions as strings.
   However, some of the extensions -- particularly the :ref:`bob.bio.video <bob.bio.video>` extension -- requires the read and write functions to handle files of type :py:class:`bob.io.base.HDF5File`.

If you plan to write your own tools, please assure that you are following the following structure.


.. _bob.bio.base.preprocessors:

Preprocessors
~~~~~~~~~~~~~

All preprocessor classes are derived from :py:class:`bob.bio.base.preprocessor.Preprocessor`.
All of them implement the following two functions:

* ``__init__(self, <parameters>)``: Initializes the preprocessing algorithm with the parameters it needs.
  The base class constructor is called in the derived class constructor, e.g. as ``bob.bio.base.preprocessor.Preprocessor.__init__(self, ...)``.
* ``__call__(self, original_data, annotations) -> data``: preprocesses the data given the dictionary of annotations (e.g. ``{'reye' : [re_y, re_x], 'leye': [le_y, le_x]}`` for face images).

  .. note::
     When the database does not provide annotations, the ``annotations`` parameter might be ``None``.

By default, the data returned by the preprocessor is of type :py:class:`numpy.ndarray`.
In that case, the base class IO functionality can be used.
If a class returns data that is **not** of type :py:class:`numpy.ndarray`, it overwrites further functions from :py:class:`bob.bio.base.preprocessor.Preprocessor` that define the IO of your class:

* ``write_data(data, data_file)``: Writes the given data (that has been generated using the ``__call__`` function of this class) to file.
* ``read_data(data_file)``: Reads the preprocessed data from file.

By default, the original data is read by :py:func:`bob.io.base.load`.
Hence, data is given as :py:class:`numpy.ndarray`\s.
When a different IO for the original data is required (for example to read videos in :py:class:`bob.bio.video.preprocessor.Video`), the following function is overridden:

* ``read_original_data(filename)``: Reads the original data from file.


.. _bob.bio.base.extractors:

Extractors
~~~~~~~~~~

Feature extractors should be derived from the :py:class:`bob.bio.base.extractor.Extractor` class.
All extractor classes provide at least the functions:

* ``__init__(self, <parameters>)``: Initializes the feature extraction algorithm with the parameters it needs.
  Calls the base class constructor, e.g. as ``bob.bio.base.extractor.Extractor.__init__(self, ...)`` (there are more parameters to this constructor, see below).
* ``__call__(self, data) -> feature``: Extracts the feature from the given preprocessed data.
  By default, the returned feature should be a :py:class:`numpy.ndarray`.

If features are not of type :py:class:`numpy.ndarray`, the ``write_feature`` function is overridden.
In this case, also the function to read that kind of features needs to be overridden:

* ``write_feature(self, feature, feature_file)``: Writes the feature (as returned by the ``__call__`` function) to the given file name.
* ``read_feature(self, feature_file) -> feature``: Reads the feature (as written by the ``save_feature`` function) from the given file name.

.. note::
   If the feature is of a class that contains and is written via a ``save(bob.io.base.HDF5File)`` method, the ``write_feature`` function does not need to be overridden.
   However, the ``read_feature`` function is required in this case.

If the feature extraction process requires to read a trained extractor model from file, the following function is overloaded:

* ``load(self, extractor_file)``: Loads the extractor from file.
  This function is called at least once before the ``__call__`` function is executed.

It is also possible to train the extractor model before it is used.
In this case, two things are done.
First, the ``train`` function is overridden:

* ``train(self, image_list, extractor_file)``: Trains the feature extractor with the given list of images and writes the ``extractor_file``.

Second, this behavior is registered in the ``__init__`` function by calling the base class constructor with more parameters: ``bob.bio.base.extractor.Extractor.__init__(self, requires_training=True, ...)``.
Given that the training algorithm needs to have the training data split by identity, the ``bob.bio.base.extractor.Extractor.__init__(self, requires_training=True, split_training_images_by_client = True, ...)`` is used instead.


.. _bob.bio.base.algorithms:

Algorithms
~~~~~~~~~~
The implementation of recognition algorithm is as straightforward.
All algorithms are derived from the :py:class:`bob.bio.base.algorithm.Algorithm` class.
The constructor of this class has the following options, which are selected according to the current algorithm:

* ``performs_projection``: If set to ``True``, features will be projected using the ``project`` function.
  With the default ``False``, the ``project`` function will not be called at all.
* ``requires_projector_training``: If ``performs_projection`` is enabled, this flag specifies if the projector needs training.
  If ``True`` (the default), the ``train_projector`` function will be called.
* ``split_training_features_by_client``: If the projector training needs training images split up by client identity, this flag is enabled.
  In this case, the ``train_projector`` function will receive a list of lists of features.
  If set to ``False`` (the default), the training features are given in one list.
* ``use_projected_features_for_enrollment``: If features are projected, by default (``True``) models are enrolled using the projected features.
  If the algorithm requires the original unprojected features to enroll the model, ``use_projected_features_for_enrollment=False`` is selected.
* ``requires_enroller_training``: Enables the enroller training.
  By default (``False``), no enroller training is performed, i.e., the ``train_enroller`` function is not called.

* ``multiple_model_scoring``: The way to handle scoring when models store several features.
  Set this parameter to ``None`` when you implement your own functionality to handle models from several features (see below).
* ``multiple_probe_scoring``: The way to handle scoring when models store several features.
  Set this parameter to ``None`` when you handle scoring with multiple probes with your own ``score_for_multiple_probes`` function (see below).

A recognition algorithm has to override at least three functions:

* ``__init__(self, <parameters>)``: Initializes the face recognition algorithm with the parameters it needs.
  Calls the base class constructor, e.g. as ``bob.bio.base.algorithm.Algorithm.__init__(self, ...)`` (there are more parameters to this constructor, see above).
* ``enroll(self, enroll_features) -> model``: Enrolls a model from the given vector of features (this list usually contains features from several files of one subject) and returns it.
  The returned model is either a :py:class:`numpy.ndarray` or an instance of a class that defines a ``save(bob.io.base.HDF5File)`` method.
  If neither of the two options are appropriate, a ``write_model`` function is defined (see below).
* ``score(self, model, probe) -> value``: Computes a similarity or probability score that the given probe feature and the given model stem from the same identity.

  .. note::
     When you use a distance measure in your scoring function, and lower distances represents higher probabilities of having the same identity, please return the negative distance.

Additionally, an algorithm may need to project the features before they can be used for enrollment or recognition.
In this case, (some of) the function(s) are overridden:

* ``train_projector(self, train_features, projector_file)``: Uses the given list of features and writes the ``projector_file``.

  .. warning::
     If you write this function, please assure that you use both ``performs_projection=True`` and ``requires_projector_training=True`` (for the latter, this is the default, but not for the former) during the base class constructor call in your ``__init__`` function.
     If you need the training data to be sorted by clients, please use ``split_training_features_by_client=True`` as well.
     Please also assure that you overload the ``project`` function.

* ``load_projector(self, projector_file)``: Loads the projector from the given file, i.e., as stored by ``train_projector``.
  This function is always called before the ``project``, ``enroll``, and ``score`` functions are executed.
* ``project(self, feature) -> feature``: Projects the given feature and returns the projected feature, which should either be a :py:class:`numpy.ndarray` or an instance of a class that defines a ``save(bob.io.base.HDF5File)`` method.

  .. note::
     If you write this function, please assure that you use ``performs_projection=True`` during the base class constructor call in your ``__init__`` function.

And once more, if the projected feature is not of type ``numpy.ndarray``, the following methods are overridden:

* ``write_feature(feature, feature_file)``: Writes the feature (as returned by the ``project`` function) to file.
* ``read_feature(feature_file) -> feature``: Reads and returns the feature (as written by the ``write_feature`` function).

Some tools also require to train the model enrollment functionality (or shortly the ``enroller``).
In this case, these functions are overridden:

* ``train_enroller(self, training_features, enroller_file)``: Trains the model enrollment with the list of lists of features and writes the ``enroller_file``.

  .. note::
     If you write this function, please assure that you use ``requires_enroller_training=True`` during the base class constructor call in your ``__init__`` function.

* ``load_enroller(self, enroller_file)``: Loads the enroller from file.
  This function is always called before the ``enroll`` and ``score`` functions are executed.


By default, it is assumed that both the models and the probe features are of type :py:class:`numpy.ndarray`.
If the ``score`` function expects models and probe features to be of a different type, these functions are overridden:

* ``write_model(self, model, model_file)``: writes the model (as returned by the ``enroll`` function).
* ``read_model(self, model_file) -> model``: reads the model (as written by the ``write_model`` function) from file.
* ``read_probe(self, probe_file) -> feature``: reads the probe feature from file.

  .. note::
     In many cases, the ``read_feature`` and ``read_probe`` functions are identical (if both are present).

Finally, the :py:class:`bob.bio.base.algorithm.Algorithm` class provides default implementations for the case that models store several features, or that several probe features should be combined into one score.
These two functions are:

* ``score_for_multiple_models(self, models, probe)``: In case your model store several features, **call** this function to compute the average (or min, max, ...) of the scores.
* ``score_for_multiple_probes(self, model, probes)``: By default, the average (or min, max, ...) of the scores for all probes are computed. **Override** this function in case you want different behavior.


Implemented Tools
-----------------

In this base class, only one feature extractor and some recognition algorithms are defined.
However, implementations of the base classes can be found in all of the ``bob.bio`` packages.
Here is a list of implementations:

* :ref:`bob.bio.base <bob.bio.base>` : :ref:`bob.bio.base.implemented`
* :ref:`bob.bio.face <bob.bio.face>` : :ref:`bob.bio.face.implemented`
* :ref:`bob.bio.video <bob.bio.video>` : :ref:`bob.bio.video.implemented`
* :ref:`bob.bio.gmm <bob.bio.gmm>` : :ref:`bob.bio.gmm.implemented`
* :ref:`bob.bio.csu <bob.bio.csu>` : :ref:`bob.bio.csu.implemented`

.. * :ref:`bob.bio.spear <bob.bio.spear>` : :ref:`bob.bio.spear.implemented`


.. todo:: complete this list, once the other packages are documented as well.


Databases
---------

Databases provide information about the data sets, on which the recognition algorithm should run on.
Particularly, databases come with one or more evaluation protocols, which defines, which part of the data should be used for training, enrollment and probing.
Some protocols split up the data into three different groups: a training set (aka. ``world`` group), a development set (aka. ``dev`` group) and an evaluation set (``eval``, sometimes also referred as test set).
Furthermore, some of the databases split off some data from the training set, which is used to perform a ZT score normalization.
Finally, most of the databases come with specific annotation files, which define additional information about the data, e.g., hand-labeled eye locations for face images.


Verification Database Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most of the data sets, we rely on the database interfaces from Bob_.
Particularly, all databases that are derived from the :py:class:`bob.db.verification.utils.Database` (click :ref:`here <verification_databases>` for a list of implemented databases) are supported by a special derivation of the databases from above.
For these databases, the special :py:class:`bob.bio.base.database.DatabaseBob` interface is provided, which takes the Bob_ database as parameter.
Several such databases are defined in the according packages, i.e., :ref:`bob.bio.spear <bob.bio.spear>`, :ref:`bob.bio.face <bob.bio.face>` and :ref:`bob.bio.video <bob.bio.video>`.
For Bob_'s ZT-norm databases, we provide the :py:class:`bob.bio.base.database.DatabaseBobZT` interface.

Additionally, a generic database interface, which is derived from :py:class:`bob.bio.base.database.DatabaseBobZT`, is the :py:class:`bob.bio.base.database.DatabaseFileList`.
This database interfaces with the :py:class:`bob.db.verification.filelist.Database`, which is a generic database based on file lists, implementing the :py:class:`bob.db.verification.utils.Database` interface.

Defining your own Database
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have your own database that you want to execute the recognition experiments on, you should first check if you could use the :ref:`Verifcation FileList Database <bob.db.verification.filelist>` interface by defining appropriate file lists for the training set, the model set, and the probes.
In most of the cases, the :py:class:`bob.db.verification.filelist.Database` should be sufficient to run experiments.
Please refer to the documentation :ref:`Documentation <bob.db.verification.filelist>` of this database for more instructions on how to configure this database.

In case you want to have a more complicated interface to your database, you are welcome to write your own database wrapper class.
In this case, you have to derive your class from the :py:class:`bob.bio.base.database.Database`, and provide the following functions:

* ``__init__(self, <your-parameters>, **kwargs)``: Constructor of your database interface.
  Please call the base class constructor, providing all the required parameters, e.g. by ``bob.bio.base.database.Database.__init__(self, **kwargs)``.
* ``all_files(self)``: Returns a list of all :py:class:`bob.bio.base.database.File` objects of the database.
  The list needs to be sorted by the file id (you can use the ``self.sort(files)`` function for sorting).
* ``training_files(self, step, arrange_by_client = False)``: A sorted list of the :py:class:`bob.bio.base.database.File` objects that is used for training.
  If ``arrange_by_clients`` is enabled, you might want to use the :py:meth:`bob.bio.base.database.Database.arrange_by_client` function to perform the job.
* ``model_ids(self, group = 'dev'): The ids for the models (usually, there is only one model per client and, thus, you can simply use the client ids) for the given group.
  Usually, providing ids for the group ``'dev'`` should be sufficient.
* ``client_id_from_model_id(self, model_id)``: Returns the client id for the given model id.
* ``enroll_files(self, model_id, group='dev')``: Returns the list of model :py:class:`bob.bio.base.database.File` objects for the given model id.
* ``probe_files(self, model_id=None, group='dev')``: Returns the list of probe files, the given model_id should be compared with.
  Usually, all probe files are compared with all model files.
  In this case, you can just ignore the ``model_id``.
  If the ``model_id`` is ``None``, this function is supposed to return *all* probe files for all models of the given group.

Additionally, you can define more lists that can be used for ZT score normalization.
In this case, derive you class from :py:class:`bob.bio.base.database.DatabaseZT` instead, and additionally overwrite the following functions:

* ``t_model_ids(self, group = 'dev')``: The ids for the T-Norm models for the given group.
* ``t_enroll_files(self, model_id, group='dev')``: Returns the list of model :py:class:`bob.bio.base.database.File` objects for the given T-Norm model id.
* ``z_probe_files(self, group='dev')``: Returns the list of Z-probe :py:class:`bob.bio.base.database.File` objects, with which all the models and T-Norm models are compared.

.. note:
   For a proper biometric recognition protocol, the identities from the models and the T-Norm models, as well as the Z-probes should be different.

For some protocols, a single probe consists of several features, see :ref:`bob.bio.base.algorithms` about strategies how to incorporate several probe files into one score.
If your database should provide this functionality, please overwrite:

* ``uses_probe_file_sets(self)``: Return ``True`` if the current protocol of the database provides multiple files for one probe.
* ``probe_file_sets(self, model_id=None, group='dev')``: Returns a list of lists of :py:class:`bob.bio.base.database.FileSet` objects.
* ``z_probe_file_sets(self, model_id=None, group='dev')``: Returns a list of lists of Z-probe :py:class:`bob.bio.base.database.FileSet` objects (only needed if the base class is :py:class:`bob.bio.base.database.DatabaseZT`).



.. _bob.bio.base.configuration-files:

Configuration Files
-------------------

One important aspect of the ``bob.bio`` packages is reproducibility.
To be able to reproduce an experiment, it is required that all parameters of all tools are present.

In ``bob.bio`` this is achieved by providing these parameters in configuration files.
In these files, an *instance* of one of the tools is generated, and assigned to a variable with a specific name.
These variable names are:

* ``database`` for an instance of a (derivation of a) :py:class:`bob.bio.base.database.Database`
* ``preprocessor`` for an instance of a (derivation of a) :py:class:`bob.bio.base.preprocessor.Preprocessor`
* ``extractor`` for an instance of a (derivation of a) :py:class:`bob.bio.base.extractor.Extractor`
* ``algorithm`` for an instance of a (derivation of a) :py:class:`bob.bio.base.algorithm.Algorithm`
* ``grid`` for an instance of the :py:class:`bob.bio.base.grid.Grid`

For example, the configuration file for a PCA algorithm, which uses 80% of variance and a cosine distance function, could look somewhat like:

.. code-block:: py

   import bob.bio.base
   import scipy.spatial

   algorithm = bob.bio.base.algorithm.PCA(subspace_dimension = 0.8, distance_function = scipy.spatial.distance.cosine, is_distance_function = True)

Some default configuration files can be found in the ``bob/bio/*/config`` directories of all ``bob.bio`` packages, but you can create configuration files in any directory you like.
In fact, since all tools have a different keyword, you can define a complete experiment in a single configuration file.


.. _bob.bio.base.resources:

Resources
---------

Finally, some of the configuration files, which sit in the ``bob/bio/*/config`` directories, are registered as *resources*.
This means that a resource is nothing else than a short name for a registered instance of one of the tools (database, preprocessor, extractor, algorithm or grid configuration) of ``bob.bio``, which has a pre-defined set of parameters.

The process of registering a resource is relatively easy.
We use the SetupTools_ mechanism of registering so-called entry points in the ``setup.py`` file of the according ``bob.bio`` package.
Particularly, we use a specific list of entry points, which are:

* ``bob.bio.database`` to register an instance of a (derivation of a) :py:class:`bob.bio.base.database.Database`
* ``bob.bio.preprocessor`` to register an instance of a (derivation of a) :py:class:`bob.bio.base.preprocessor.Preprocessor`
* ``bob.bio.extractor`` to register an instance of a (derivation of a) :py:class:`bob.bio.base.extractor.Extractor`
* ``bob.bio.algorithm`` to register an instance of a (derivation of a) :py:class:`bob.bio.base.algorithm.Algorithm`
* ``bob.bio.grid`` to register an instance of the :py:class:`bob.bio.base.grid.Grid`

For each of the tools, several resources are defined, which you can list with the ``./bin/resources.py`` command line.

When you want to register your own resource, make sure that your configuration file is importable (usually it is sufficient to have an empty ``__init__.py`` file in the same directory as your configuration file).
Then, you can simply add a line inside the according ``entry_points`` section of the ``setup.py`` file (you might need to create that section, just follow the example of the ``setup.py`` file that you can find online in the base directory of our `bob.bio.base GitHub page <http://github.com/bioidiap/bob.bio.base>`__).

After re-running ``./bin/buildout``, your new resource should be listed in the output of ``./bin/resources.py``.


.. include:: links.rst
