.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: Mon 26 Sep 2022 10:35:22 UTC+02

.. _ bob.bio.base.database_interface:

====================
 Database Interface
====================

Here is the description of how to feed data to a bob biometric pipeline.

Structure
=========

The database for biometric experiments need a slightly more complex structure as `the
default from bob.pipelines <bob.pipelines.csv_database>`_.
On top of the *protocol* and *group* classification, we must add a *purpose* "layer".
Each sub-pipeline will take a different list of samples, like an *enrollment* set of
samples and a *probe* set, for each group in each protocol.

To do so, the structure of CSV files is different:

.. code-block::

    atnt_database
     |
     +- default
         |
         +- train
         |   |
         |   +- for_background_model.csv
         |
         +- dev
         |   |
         |   +- for_enrolling.csv
         |   +- for_probing.csv
         |   +- for_matching.csv
         |
         +- test
             |
             +- for_enrolling.csv
             +- for_probing.csv
             +- for_matching.csv

In this database definition of ``atnt_database`` we have one protocol ``default``,
three groups ``train``, ``dev``, and ``eval`` which contain different set of samples
for enrollment and scoring (except train that has just one pool of samples in this
case).

- The ``for_enrolling.csv`` files define the creation of samples for enrollment
  purposes. Those samples will create the enrolled references.
- The ``for_probing.csv`` files dictate to create samples for probes that will be
  compared against the enrolled references.
- The optional ``for_matching.csv`` files contains the list of comparisons that will
  be made between samples in ``for_enrolling.csv`` and ``for_probing.csv``. If this
  file is not present, all the probes will be compared against all the references.

File format
===========

All files should follow the CSV format with a header. Here is an example of the
``for_enrolling.csv`` file:

.. code-block::

    path,template_id,metadata1,metadata2
    dev/u01/s01.png,u01s01,male,29
    dev/u01/s02.png,u01s02,male,29
    dev/u02/s01.png,u02s01,female,32
    [...]

Note that the paths should be relative to the ``original_directory``
