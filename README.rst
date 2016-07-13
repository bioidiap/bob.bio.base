.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Thu 30 Jan 08:46:53 2014 CET

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.bio.base/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.bio.base/master/index.html
.. image:: http://travis-ci.org/bioidiap/bob.bio.base.svg?branch=v2.0.10
   :target: https://travis-ci.org/bioidiap/bob.bio.base?branch=v2.0.10
.. image:: https://coveralls.io/repos/bioidiap/bob.bio.base/badge.svg?branch=v2.0.10
   :target: https://coveralls.io/r/bioidiap/bob.bio.base?branch=v2.0.10
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/bioidiap/bob.bio.base/tree/master
.. image:: http://img.shields.io/pypi/v/bob.bio.base.png
   :target: https://pypi.python.org/pypi/bob.bio.base
.. image:: http://img.shields.io/pypi/dm/bob.bio.base.png
   :target: https://pypi.python.org/pypi/bob.bio.base

==================================================
 Scripts to run biometric recognition experiments
==================================================

This package is part of the ``bob.bio`` packages, which allow to run comparable and reproducible biometric recognition experiments on publicly available databases.

This package contains basic functionality to run biometric recognition experiments.
It provides a generic ``./bin/verify.py`` script that takes several parameters, including:

* A database and its evaluation protocol
* A data preprocessing algorithm
* A feature extraction algorithm
* A biometric recognition algorithm

All these steps of the biometric recognition system are given as configuration files.

In this base class implementation, only a few algorithms (such as PCA, LDA, PLDA, BIC) are implemented, while most algorithms that are more specialized are provided by other packages, which are usually in the ``bob.bio`` namespace, such as:

* `bob.bio.spear <http://pypi.python.org/pypi/bob.bio.spear>`__ for speaker recognition recognition databases, features and algorithms
* `bob.bio.face <http://pypi.python.org/pypi/bob.bio.face>`__ for face recognition databases, features and algorithms
* `bob.bio.video <http://pypi.python.org/pypi/bob.bio.video>`__ for video-based databases and algorithms
* `bob.bio.gmm <http://pypi.python.org/pypi/bob.bio.gmm>`__ for Gaussian-mixture-model-based algorithms
* `bob.bio.csu <http://pypi.python.org/pypi/bob.bio.csu>`__ for wrapper classes of the `CSU Face Recognition Resources <http://www.cs.colostate.edu/facerec>`__ (see `Installation Instructions <http://pythonhosted.org/bob.bio.csu/installation.html>`__ of ``bob.bio.csu``).


Moreover, a script for score fusion ``./bin/fusion_llr.py`` is provided to do score-level fusion using Logistic regression to combine outputs from different systems.

Additionally, a generic script ``./bin/evaluate.py`` is provided that can generate several types of plots (such as ROC, DET and CMC curves) and compute several measures (such as HTER, Cllr) to evaluate your experiments.


Installation
------------
To create your own working package using one or more of the ``bob.bio`` packages, please follow the `Installation Instructions <http://pythonhosted.org/bob.bio.base/installation.html>`__ of the ``bob.bio`` packages.

To install this package -- alone or together with other `Packages of Bob <https://github.com/idiap/bob/wiki/Packages>`_ -- please read the `Installation Instructions <https://github.com/idiap/bob/wiki/Installation>`__.
For Bob_ to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.

Documentation
-------------
For further documentation on this package, please read the `Stable Version <http://pythonhosted.org/bob.bio.base/index.html>`_ or the `Latest Version <https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.bio.base/master/index.html>`_ of the documentation.
For a list of tutorials on this or the other packages ob Bob_, or information on submitting issues, asking questions and starting discussions, please visit its website.

.. _bob: https://www.idiap.ch/software/bob
