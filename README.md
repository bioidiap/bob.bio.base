[![badge doc](https://img.shields.io/badge/docs-latest-orange.svg)](https://www.idiap.ch/software/bob/docs/bob/bob.bio.base/master/sphinx/index.html)
[![badge pipeline](https://gitlab.idiap.ch/bob/bob.bio.base/badges/master/pipeline.svg)](https://gitlab.idiap.ch/bob/bob.bio.base/commits/master)
[![badge coverage](https://gitlab.idiap.ch/bob/bob.bio.base/badges/master/coverage.svg)](https://www.idiap.ch/software/bob/docs/bob/bob.bio.base/master/coverage)
[![badge gitlab](https://img.shields.io/badge/gitlab-project-0000c0.svg)](https://gitlab.idiap.ch/bob/bob.bio.base)

# Tools to run biometric recognition experiments

This package is part of the signal-processing and machine learning toolbox
[Bob](https://www.idiap.ch/software/bob). It provides tools to run comparable
and reproducible biometric recognition experiments on publicly available
databases.

The [User Guide](https://www.idiap.ch/software/bob/docs/bob/bob.bio.base/master/sphinx/index.html)
provides installation and usage instructions.
If you run biometric recognition experiments using the bob.bio framework,
please cite at least one of the following in your scientific publication:

``` bibtext
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
```

## Installation

Complete Bob's
[installation instructions](https://www.idiap.ch/software/bob/install). Then, to
install this package, run:

``` sh
conda install bob.bio.base
```

## Contact

For questions or reporting issues to this software package, contact our
development [mailing list](https://www.idiap.ch/software/bob/discuss).
