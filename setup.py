#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import dist, setup

dist.Distribution(dict(setup_requires=["bob.extension"]))

from bob.extension.utils import find_packages, load_requirements

install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(
    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name="bob.bio.base",
    version=open("version.txt").read().rstrip(),
    description="Tools for running biometric recognition experiments",
    url="https://gitlab.idiap.ch/bob/bob.bio.base",
    license="BSD",
    author="Manuel Gunther",
    author_email="siebenkopf@googlemail.com",
    keywords="bob, biometric recognition, evaluation",
    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description=open("README.rst").read(),
    # This line is required for any distutils based packaging.
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires=install_requires,
    # Your project should be called something like 'bob.<foo>' or
    # 'bob.<foo>.<bar>'. To implement this correctly and still get all your
    # packages to be imported w/o problems, you need to implement namespaces
    # on the various levels of the package and declare them here. See more
    # about this here:
    # http://peak.telecommunity.com/DevCenter/setuptools#namespace-packages
    #
    # Our database packages are good examples of namespace implementations
    # using several layers. You can check them out here:
    # https://www.idiap.ch/software/bob/packages
    # This entry defines which scripts you will have inside the 'bin' directory
    # once you install the package (or run 'bin/buildout'). The order of each
    # entry under 'console_scripts' is like this:
    #   script-name-at-bin-directory = module.at.your.library:function
    #
    # The module.at.your.library is the python file within your library, using
    # the python syntax for directories (i.e., a '.' instead of '/' or '\').
    # This syntax also omits the '.py' extension of the filename. So, a file
    # installed under 'example/foo.py' that contains a function which
    # implements the 'main()' function of particular script you want to have
    # should be referred as 'example.foo:main'.
    #
    # In this simple example we will create a single program that will print
    # the version of bob.
    entry_points={
        # scripts should be declared using this entry:
        "console_scripts": [
            "resources.py      = bob.bio.base.script.resources:resources",
        ],
        "bob.bio.config": [
            "dummy             = bob.bio.base.test.dummy.config",  # for test purposes only
            "dummy2            = bob.bio.base.test.dummy.config2",  # for test purposes only
            "dummy_samples     = bob.bio.base.test.dummy.samples_list",  # for test purposes only
            "atnt              = bob.bio.base.config.database.atnt",
        ],
        "bob.bio.database": [
            "dummy             = bob.bio.base.test.dummy.database:database",  # for test purposes only
            "atnt              = bob.bio.base.config.database.atnt:database",
        ],
        # main entry for bob bio cli
        "bob.cli": [
            "bio               = bob.bio.base.script.bio:bio",
            "vulnerability     = bob.bio.base.script.vulnerability:vulnerability",
        ],
        # bob bio scripts
        "bob.bio.cli": [
            "annotate          = bob.bio.base.script.annotate:annotate",
            "annotate-samples  = bob.bio.base.script.annotate:annotate_samples",
            "metrics           = bob.bio.base.script.commands:metrics",
            "multi-metrics     = bob.bio.base.script.commands:multi_metrics",
            "roc               = bob.bio.base.script.commands:roc",
            "det               = bob.bio.base.script.commands:det",
            "epc               = bob.bio.base.script.commands:epc",
            "hist              = bob.bio.base.script.commands:hist",
            "cmc               = bob.bio.base.script.commands:cmc",
            "dir               = bob.bio.base.script.commands:dir",
            "gen               = bob.bio.base.script.gen:gen",
            "evaluate          = bob.bio.base.script.commands:evaluate",
            "sort              = bob.bio.base.script.sort:sort",
            "pipeline         = bob.bio.base.script.pipeline:pipeline",
            "compare-samples   = bob.bio.base.script.compare_samples:compare_samples",
        ],
        # annotators
        "bob.bio.annotator": [
            "dummy             = bob.bio.base.test.dummy.annotator:annotator",
        ],
        # run pipelines
        "bob.bio.pipeline.cli": [
            "simple = bob.bio.base.script.pipeline_simple:pipeline_simple",
            "score-norm = bob.bio.base.script.pipeline_score_norm:pipeline_score_norm",
            "transform = bob.bio.base.script.pipeline_transform:pipeline_transform",
            "train = bob.bio.base.script.pipeline_train:pipeline_train",
        ],
        # Vulnerability analysis commands
        "bob.vuln.cli": [
            "metrics           = bob.bio.base.script.vuln_commands:metrics",
            "hist              = bob.bio.base.script.vuln_commands:hist",
            "det               = bob.bio.base.script.vuln_commands:det",
            "roc               = bob.bio.base.script.vuln_commands:roc",
            "epc               = bob.bio.base.script.vuln_commands:epc",
            "epsc              = bob.bio.base.script.vuln_commands:epsc",
            "gen               = bob.bio.base.script.vuln_commands:gen",
            "fmr_iapmr         = bob.bio.base.script.vuln_commands:fmr_iapmr",
            "evaluate          = bob.bio.base.script.vuln_commands:evaluate",
        ],
    },
    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers=[
        "Framework :: Bob",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
