.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base:

=====================================
 Resources for biometric experiments
=====================================


``bob.bio.base`` provides open-source tools to run comparable and reproducible biometric recognition experiments.
It covers the following biometrics traits:

 * Face Biometrics: `bob.bio.face <http://gitlab.idiap.ch/bob/bob.bio.face>`__
 * Vein Biometrics: `bob.bio.vein <http://gitlab.idiap.ch/bob/bob.bio.vein>`__
 * Speaker Biometrics: `bob.bio.spear <http://gitlab.idiap.ch/bob/bob.bio.spear>`__


Get Started
============

This package defines the structure of biometric experiments. After installing the necessary environment, you can try out a simple comparison between two (or more) samples using a face recognition algorithm from `bob.bio.face <http://gitlab.idiap.ch/bob/bob.bio.face>`__, for example. Run the following command::

  $ bob bio compare-samples --pipeline facenet-sanderberg me.png not_me.png

The ``--pipeline`` option indicates which algorithm should be used to compare the pictures.
The list of all available pipelines is available in the help text of the ``--pipeline`` option::

  $ bob bio compare-samples --help
    ...
    Options:
      -p, --pipeline CUSTOM     Vanilla biometrics pipeline composed of a scikit-
                                learn Pipeline and a BioAlgorithm Can be a
                                ``bob.bio.pipeline`` entry point, a module name,
                                or a path to a Python file which contains a
                                variable named `pipeline`.Available entry points
                                are: ..., facenet-sanderberg, ...


Of course, with that command, you can run every possible biometric experiment by *headbutting* the problem and executing everything by hand.
Or you could use the tools that we offer here to set up an :ref:`experimentation pipeline <bob.bio.base.build_pipelines>`, structure your data within a :ref:`database interface <bob.bio.base.database_interface>` and run a whole experiment in one swoop.



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

   biometrics_intro
   pipeline_simple_intro
   database_interface
   pipeline_simple_features
   pipeline_score_norm
   annotators
   legacy
   vulnerability_analysis



Reference Manual
================

.. toctree::
   :maxdepth: 2

   py_api


References
==========

.. [Auckenthaler2000] Auckenthaler, Roland, Michael Carey, and Harvey Lloyd-Thomas. "Score normalization for text-independent speaker verification systems." Digital Signal Processing 10.1 (2000): 42-54.
.. [Mariethoz2005] Mariethoz, Johnny, and Samy Bengio. "A unified framework for score normalization techniques applied to text-independent speaker verification." IEEE signal processing letters 12.7 (2005): 532-535.
.. [Mandasari2014] `Mandasari, Miranti Indar, et al. "Score calibration in face recognition." Iet Biometrics 3.4 (2014): 246-256.`



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. include:: links.rst
