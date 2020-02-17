.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base:

===========================================
 Running Biometric Recognition Experiments
===========================================

The ``bob.bio`` packages provide open source tools to run comparable and reproducible biometric recognition experiments.
To design a biometric recognition experiment, you must choose:

* A database to use for the raw biometric data and a protocol that defines how to use that data,
* A data preprocessing algorithm to clean up the raw biometric data,
* A feature extractor to extract the desired type of features from the preprocessed data,
* A biometric matching algorithm,
* An evaluation method to make sense of the matching scores.

The ``bob.bio`` packages contain several implementations of each of the above steps, so you can either choose from the existing methods or use your own.

.. note::
   The ``bob.bio`` packages are derived from the former `FaceRecLib <http://pypi.python.org/pypi/facereclib>`__, which is herewith outdated.

Structure of the Biometric Recognition Framework
================================================

The :py:mod:`bob.bio.base` package includes the basic definition of a biometric recognition experiment, as well as a generic script, which can execute the full biometric experiment in a single command line.
Changing the employed tools, such as the database, protocol, preprocessor, feature extractor or matching algorithm is as simple as changing a parameter in a configuration file or on the command line.

The implementation of (most of) the tools is separated into other packages in the ``bob.bio`` namespace.
All of these packages can be easily combined.
Here is a growing list of derived packages:

* :ref:`bob.bio.spear <bob.bio.spear>` Tools to run speaker recognition experiments, including voice activity detection, Cepstral feature extraction, and speaker databases
* :ref:`bob.bio.vein <bob.bio.vein>` Tools to run vein recognition experiments, such as finger RoI detection, image binarization and template matching, and access to multiple vein image databases
* :ref:`bob.bio.face <bob.bio.face>` Tools to run face recognition experiments, such as face detection, facial feature extraction and comparison, and face image databases
* :ref:`bob.bio.video <bob.bio.video>` An extension of face recognition algorithms to run on video data, and the according video databases
* :ref:`bob.bio.gmm <bob.bio.gmm>` Algorithms based on Gaussian Mixture Modeling (GMM) such as Inter-Session Variability modeling (ISV) or Total Variability modeling (TV, aka. I-Vector) [Pri07]_ and [ESM+13]_.


Citing our Publications
=======================

If you run biometric recognition experiments using the bob.bio framework, please cite at least one of the following in your scientific publication:

.. code-block:: tex

  @inbook{guenther2016face,
    chapter = {Face Recognition in Challenging Environments: An Experimental and Reproducible Research Survey},
    author = {G\"unther, Manuel and El Shafey, Laurent and Marcel, S\'ebastien},
    editor = {Bourlai, Thirimachos},
    title = {Face Recognition Across the Imaging Spectrum},
    edition = {1},
    year = {2016},
    month = feb,
    publisher = {Springer}
  }

  @inproceedings{guenther2012facereclib,
    title = {An Open Source Framework for Standardized Comparisons of Face Recognition Algorithms},
    author = {G\"unther, Manuel and Wallace, Roy and Marcel, S\'ebastien},
    editor = {Fusiello, Andrea and Murino, Vittorio and Cucchiara, Rita},
    booktitle = {European Conference on Computer Vision (ECCV) Workshops and Demonstrations},
    series = {Lecture Notes in Computer Science},
    volume = {7585},
    year = {2012},
    month = oct,
    pages = {547-556},
    publisher = {Springer},
  }


Users Guide
===========

.. toctree::
   :maxdepth: 2

   installation
   struct_bio_rec_sys
   experiments
   implementation
   filelist-guide
   more
   annotations
   openbr


Reference Manual
================

.. toctree::
   :maxdepth: 2

   implemented
   py_api



References
==========

.. [TP91]    *M. Turk and A. Pentland*. **Eigenfaces for recognition**. Journal of Cognitive Neuroscience, 3(1):71-86, 1991.
.. [ZKC+98]  *W. Zhao, A. Krishnaswamy, R. Chellappa, D. Swets and J. Weng*. **Discriminant analysis of principal components for face recognition**, pages 73-85. Springer Verlag Berlin, 1998.
.. [Pri07]   *S. J. D. Prince*. **Probabilistic linear discriminant analysis for inferences about identity**. Proceedings of the International Conference on Computer Vision. 2007.
.. [ESM+13]  *L. El Shafey, Chris McCool, Roy Wallace and Sébastien Marcel*. **A scalable formulation of probabilistic linear discriminant analysis: applied to face recognition**. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(7):1788-1794, 7/2013.
.. [MWP98]   *B. Moghaddam, W. Wahid and A. Pentland*. **Beyond eigenfaces: probabilistic matching for face recognition**. IEEE International Conference on Automatic Face and Gesture Recognition, pages 30-35. 1998.
.. [GW09]    *M. Günther and R.P. Würtz*. **Face detection and recognition using maximum likelihood classifiers on Gabor graphs**. International Journal of Pattern Recognition and Artificial Intelligence, 23(3):433-461, 2009.


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
