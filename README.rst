.. vim: set fileencoding=utf-8 :
.. Tue 16 Aug 15:00:20 CEST 2016

.. image:: https://img.shields.io/badge/docs-v7.0.0-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.bio.base/v7.0.0/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.bio.base/badges/v7.0.0/pipeline.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.base/commits/v7.0.0
.. image:: https://gitlab.idiap.ch/bob/bob.bio.base/badges/v7.0.0/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.base/commits/v7.0.0
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.bio.base


================================================
 Tools to run biometric recognition experiments
================================================

This package is part of the signal-processing and machine learning toolbox
Bob_. It provides tools to run comparable and reproducible biometric
recognition experiments on publicly available databases.

The `User Guide`_ provides installation and usage instructions.
If you run biometric recognition experiments using the bob.bio framework, please cite at least one of the following in your scientific publication::

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

Installation
------------

Complete Bob's `installation`_ instructions. Then, to install this package,
run::

  $ conda install bob.bio.base


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
.. _user guide: https://www.idiap.ch/software/bob/docs/bob/bob.bio.base/v7.0.0/index.html
