.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. date: Wed 21 Sep 2020 15:45:00 UTC+02

..  _bob.bio.base.pipeline_score_norm:

===================
Score normalization
===================


Score normalization aims to compensate for statistical variations in output scores
due to changes in the conditions across different enrollment and probe samples.
This is achieved by scaling distributions of system output scores to better
facilitate the application of a single, global threshold for authentication.

Bob has implemented the :py:class:`bob.bio.base.pipelines.PipelineScoreNorm` which is an
extension of regular :py:class:`bob.bio.base.pipelines.PipelineSimple` where a post process
step is appended to the scoring stage.
Bob has implemented three different strategies to normalize scores with two post processors, and these strategies are presented in the next subsections.

.. warning::
  Not all databases support the score normalization operations.
  Please look below at *Score normalization and databases* for more information on how to enable score normalization in databases.

.. _znorm:

Z-Norm
======

Given a score :math:`s_i`, Z-Norm [Auckenthaler2000]_ and [Mariethoz2005]_
(zero-normalization) scales this value by the mean (:math:`\mu`) and standard
deviation (:math:`\sigma`) of an impostor score distribution. This score
distribution can be computed beforehand, and it is defined as the following.

.. math::

  zs_i = \frac{s_i - \mu}{\sigma}


This scoring technique is implemented in our API via :py:func:`bob.bio.base.pipelines.ZNormScores`.

Currently, the ZNorm is available via the following CLI command ::

 $ bob bio pipeline score-norm [SIMPLE-PIPELINE-COMMANDS] --score-normalization-type znorm


.. _tnorm:

T-Norm
======

T-norm [Auckenthaler2000]_ and [Mariethoz2005]_ (Test-normalization) operates
in a probe-centric manner.
If in the Z-Norm :math:`\mu` and :math:`\sigma` are estimated using an impostor set of models and its scores, the t-norm computes these statistics using the current probe sample against at set of models in a cohort :math:`\Theta_{c}`.
A cohort can be any semantic organization that is
sensible to your recognition task, such as sex (male and females), ethnicity,
age, etc. and is defined as the following:

.. math::

  ts_i = \frac{s_i - \mu}{\sigma}

where, :math:`s_i` is :math:`P(x_i | \Theta)` (the score given the claimed
model), :math:`\mu = \frac{ \sum\limits_{i=0}^{N} P(x_i | \Theta_{c}) }{N}`
(:math:`\Theta_{c}` are the models of one co-hort) and :math:`\sigma` is the
standard deviation computed using the same criteria used to compute
:math:`\mu`.


This scoring technique is implemented in our API via :py:func:`bob.bio.base.pipelines.TNormScores`.

Currently, the ZNorm is available via the following CLI command ::

  bob bio pipeline score-norm [SIMPLE-PIPELINE-COMMANDS] --score-normalization-type tnorm


.. note::

  T-norm introduces extra computation during scoring, as the probe samples
  need to be compared to each cohort model in order to have :math:`\mu` and
  :math:`\sigma`.

S-Norm
======

.. todo::

  To be implemented


Score normalization and databases
=================================
.. _score_norm_databases:

To enable the above mentioned score normalization strategies it is assumed that
you passed through this :ref:`section <bob.bio.base.database.csv_file_structure>`.
Once you have absorbed that, enabling score normalization operations to your database is easy.
It consists of adding the following files in bold at the CSV database file
structure:

.. code-block:: text

  my_dataset
  |
  +-- my_protocol_1
      |
      +-- norm
      |    |
      |    +-- train_world.csv
      |    +-- *for_tnorm.csv*
      |    +-- *for_znorm.csv*
      |
      +-- dev
      |   |
      |   +-- for_models.csv
      |   +-- for_probes.csv
      |
      +-- eval
           |
           +-- for_models.csv
           +-- for_probes.csv


.. todo::

  This is no longer up to date!

The file format is identical as in the current :ref:`CSV interface <bob.bio.base.database.csv_file_interface>`



====================
Calibration by group
====================

  Implements an adaptation of the Categorical Calibration defined in [Mandasari2014]_.

.. todo::
  Complete this section

.. todo::

  Discuss all the four calibration strategies
