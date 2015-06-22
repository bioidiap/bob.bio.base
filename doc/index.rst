.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base:

===========================================
 Running Biometric Recognition Experiments
===========================================

The ``bob.bio`` packages provide open source tools to run comparable and reproducible biometric recognition experiments.
To design a biometric recognition experiment, one has to choose:

* a databases containing the original data, and a protocol that defines how to use the data,
* a data preprocessing algorithm, i.e., face detection for face recognition experiments or voice activity detection for speaker recognition
* the type of features to extract from the preprocessed data,
* the biometric recognition algorithm to employ, and
* the way to evaluate the results

For any of these parts, several different types are implemented in the ``bob.bio`` packages, and basically any combination of the five parts can be executed.
For each type, several meta-parameters can be tested.
This results in a nearly infinite amount of possible experiments that can be run using the current setup.
But it is also possible to use your own database, preprocessing, feature type, or biometric recognition algorithm and test this against the baseline algorithms implemented in the our packages.

The ``bob.bio`` packages derived from the former `FaceRecLib <http://pypi.python.org/pypi/facereclib>`__, which is herewith outdated.

This package :py:mod:`bob.bio.base` includes the basic definition of a biometric recognition experiment, as well as a generic script, which can execute the full biometric experiment in a single command line.
Changing the employed tolls such as the database, protocol, preprocessor, feature extractor or recognition algorithm is as simple as changing a command line parameter.

The implementation of (most of) the tools is separated into other packages in the ``bob.bio`` namespace.
All these packages can be easily combined.
Here is a growing list of derived packages:

* :ref:`bob.bio.spear <bob.bio.spear>` Tools to run speaker recognition experiments, including voice activity detection, Cepstral feature extraction, and speaker databases
* :ref:`bob.bio.face <bob.bio.face>` Tools to run face recognition experiments, such as face detection, facial feature extraction and comparison, and face image databases
* :ref:`bob.bio.video <bob.bio.video>` An extension of face recognition algorithms to run on video data, and the according video databases
* :ref:`bob.bio.gmm <bob.bio.gmm>` Algorithms based on Gaussian Mixture Modeling (GMM) such as Inter-Session Variability modeling (ISV) or Total Variability modeling (TV, aka. I-Vector)
* :ref:`bob.bio.csu <bob.bio.csu>` Wrapper classes for the `CSU Face Recognition Resources <http://www.cs.colostate.edu/facerec>`_ to be run with ``bob.bio``.

If you are interested, please continue reading:


===========
Users Guide
===========

.. toctree::
   :maxdepth: 2

   installation
   experiments
   implementation
   implemented
   py_api
..   evaluate

================
Reference Manual
================

.. toctree::
   :maxdepth: 2

   manual_databases
   manual_preprocessors
   manual_features
   manual_tools
   manual_utils


ToDo-List
=========

This documentation is still under development.
Here is a list of things that needs to be done:

.. todolist::


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
