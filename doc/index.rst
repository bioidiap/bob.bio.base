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
* a data preprocessing algorithm, i.e., face detection for face recognition experiments or voice activity detection for speaker recognition,
* the type of features to extract from the preprocessed data,
* the biometric recognition algorithm to employ,
* the score fusion to combine outputs from different systems, and
* the way to evaluate the results

For any of these parts, several different types are implemented in the ``bob.bio`` packages, and basically any combination of the five parts can be executed.
For each type, several meta-parameters can be tested.
This results in a nearly infinite amount of possible experiments that can be run using the current setup.
But it is also possible to use your own database, preprocessor, feature extractor, or biometric recognition algorithm and test this against the baseline algorithms implemented in the our packages.

.. note::
   The ``bob.bio`` packages are derived from the former `FaceRecLib <http://pypi.python.org/pypi/facereclib>`__, which is herewith outdated.

This package :py:mod:`bob.bio.base` includes the basic definition of a biometric recognition experiment, as well as a generic script, which can execute the full biometric experiment in a single command line.
Changing the employed tolls such as the database, protocol, preprocessor, feature extractor or recognition algorithm is as simple as changing a command line parameter.

The implementation of (most of) the tools is separated into other packages in the ``bob.bio`` namespace.
All these packages can be easily combined.
Here is a growing list of derived packages:

* :ref:`bob.bio.spear <bob.bio.spear>` Tools to run speaker recognition experiments, including voice activity detection, Cepstral feature extraction, and speaker databases
* :ref:`bob.bio.face <bob.bio.face>` Tools to run face recognition experiments, such as face detection, facial feature extraction and comparison, and face image databases
* :ref:`bob.bio.video <bob.bio.video>` An extension of face recognition algorithms to run on video data, and the according video databases
* :ref:`bob.bio.gmm <bob.bio.gmm>` Algorithms based on Gaussian Mixture Modeling (GMM) such as Inter-Session Variability modeling (ISV) or Total Variability modeling (TV, aka. I-Vector)
* `bob.bio.csu <http://pypi.python.org/pypi/bob.bio.csu>`__ for wrapper classes of the `CSU Face Recognition Resources <http://www.cs.colostate.edu/facerec>`__ (see `Installation Instructions <http://pythonhosted.org/bob.bio.csu/installation.html>`__ of ``bob.bio.csu``).

If you are interested, please continue reading:


===========
Users Guide
===========

.. toctree::
   :maxdepth: 2

   installation
   experiments
   implementation
   more

================
Reference Manual
================

.. toctree::
   :maxdepth: 2

   implemented
   py_api


==========
References
==========

.. [TP91]    *M. Turk and A. Pentland*. **Eigenfaces for recognition**. Journal of Cognitive Neuroscience, 3(1):71-86, 1991.
.. [ZKC+98]  *W. Zhao, A. Krishnaswamy, R. Chellappa, D. Swets and J. Weng*. **Discriminant analysis of principal components for face recognition**, pages 73-85. Springer Verlag Berlin, 1998.
.. [Pri07]   *S. J. D. Prince*. **Probabilistic linear discriminant analysis for inferences about identity**. Proceedings of the International Conference on Computer Vision. 2007.
.. [ESM+13]  *L. El Shafey, Chris McCool, Roy Wallace and Sébastien Marcel*. **A scalable formulation of probabilistic linear discriminant analysis: applied to face recognition**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1788-1794, 7/2013.
.. [MWP98]   *B. Moghaddam, W. Wahid and A. Pentland*. **Beyond eigenfaces: probabilistic matching for face recognition**. IEEE International Conference on Automatic Face and Gesture Recognition, pages 30-35. 1998.
.. [GW09]    *M. Günther and R.P. Würtz*. **Face detection and recognition using maximum likelihood classifiers on Gabor graphs**. International Journal of Pattern Recognition and Artificial Intelligence, 23(3):433-461, 2009.


=========
ToDo-List
=========

This documentation is still under development.
Here is a list of things that needs to be done:

.. todolist::


==================
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
