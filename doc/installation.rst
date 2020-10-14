.. vim: set fileencoding=utf-8 :
.. author: Manuel Günther <manuel.guenther@idiap.ch>
.. author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>
.. author: Yannick Dayer <yannick.dayer@idiap.ch>
.. date: Thu Sep 20 11:58:57 CEST 2012

.. _bob.bio.base.installation:

=========================
Installation Instructions
=========================

In order to run experiments with the ``bob.bio`` packages, you have to configure

The recommended way to install a bob_ package is by using :py:mod:`bob.devtools` (bdt).
In this section, everything you need to install will be layed out, but for more details, you should follow the instructions in the :py:mod:`bob.devtools` documentation `here <https://www.idiap.ch/software/bob/docs/bob/bob.devtools/master/install.html>`__.

conda_ (or miniconda) and `git <https://git-scm.com/>`__ need to be installed.

Prerequisite: Create a bdt environment
--------------------------------------

If you don't have an environment containing the :py:mod:`bob.devtools` utility, you should create a ``bdt`` environment::

$ conda create -n bdt -c https://www.idiap.ch/software/bob/conda bob bob.devtools


Then before creating the development environment, you must activate the ``bdt`` environment::

$ conda activate bdt

After this, you can proceed to the installation of the environment specific to developing a package.


Create your development environment
-----------------------------------

You must first have the source of :py:mod:`bob.bio.base`. You can fetch it as a git repository::

$ git clone https://gitlab.idiap.ch/bob/bob.bio.base
$ cd bob.bio.base

To work on :py:mod:`bob.bio.base` it is recommended to have a dedicated environment, preventing conflicts of version with different packages, and ensuring that needed dependencies are installed.
To create this conda_ environment, ensure that your ``bdt`` conda environment is activated, and run::

$ bdt create -vv dev

This will create a ``dev`` conda environment. You can proceed to activate this environment, but should first deactivate the ``bdt`` environment::

$ conda deactivate
$ conda activate dev


Build the executables
---------------------

This step will finally create the commands and executables that you need to run anything in bob.bio.base. For that, we use buildout_. (Make sure that you are still in the bob.bio.base directory you checked out earlier, and your conda development environment is active)::

$ buildout

This will create a ``bin`` folder containing the executables, all linked correctly to the development environment. This folder contains notably:

- **The bob executable**: This is the main entry point of your *bob* commands::

  $ bin/bob bio --help

- **A python executable**: Use it to run quick experiments in command line or to execute scripts::

  $ bin/python

- **Nosetests**: This is a test utility that certifies that everything is installed correctly::

  $ bin/nosetests bob.bio.base

- **Sphinx utilities**: Used to build the documentation::

  $ bin/sphinx-build doc doc/build/html


.. include:: links.rst
