#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Mon 16 Apr 08:18:08 2012 CEST

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

    # This is the basic information about your project. Modify all this
    # information before releasing code publicly.
    name = 'bob.bio.base',
    version = open("version.txt").read().rstrip(),
    description = 'Tools for running biometric recognition experiments',

    url = 'https://gitlab.idiap.ch/bob/bob.bio.base',
    license = 'BSD',
    author = 'Manuel Gunther',
    author_email = 'siebenkopf@googlemail.com',
    keywords = 'bob, biometric recognition, evaluation',

    # If you have a better, long description of your package, place it on the
    # 'doc' directory and then hook it here
    long_description = open('README.rst').read(),

    # This line is required for any distutils based packaging.
    packages = find_packages(),
    include_package_data = True,
    zip_safe=False,

    # This line defines which packages should be installed when you "install"
    # this package. All packages that are mentioned here, but are not installed
    # on the current system will be installed locally and only visible to the
    # scripts of this package. Don't worry - You won't need administrative
    # privileges when using buildout.
    install_requires = install_requires,

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
    entry_points = {

      # scripts should be declared using this entry:
      'console_scripts' : [
        'verify.py         = bob.bio.base.script.verify:main',
        'resources.py      = bob.bio.base.script.resources:resources',
        'databases.py      = bob.bio.base.script.resources:databases',
        'evaluate.py       = bob.bio.base.script.evaluate:main',
        'collect_results.py = bob.bio.base.script.collect_results:main',
        'grid_search.py    = bob.bio.base.script.grid_search:main',
        'preprocess.py     = bob.bio.base.script.preprocess:main',
        'extract.py        = bob.bio.base.script.extract:main',
        'enroll.py         = bob.bio.base.script.enroll:main',
        'score.py          = bob.bio.base.script.score:main',
        'fuse_scores.py    = bob.bio.base.script.fuse_scores:main',
      ],

      'bob.bio.config': [
        'dummy             = bob.bio.base.test.dummy.config', # for test purposes only
        'dummy2            = bob.bio.base.test.dummy.config2', # for test purposes only
      ],

      'bob.bio.database': [
        'dummy             = bob.bio.base.test.dummy.database:database', # for test purposes only
      ],

      'bob.bio.preprocessor': [
        'dummy             = bob.bio.base.test.dummy.preprocessor:preprocessor', # for test purposes only
        'filename          = bob.bio.base.config.preprocessor.filename:preprocessor',
        'dummy_metadata    = bob.bio.base.test.dummy.preprocessor:preprocessor_metadata',
      ],

      'bob.bio.extractor': [
        'dummy             = bob.bio.base.test.dummy.extractor:extractor', # for test purposes only
        'dummy_metadata    = bob.bio.base.test.dummy.extractor:extractor_metadata', # for test purposes only
        'linearize         = bob.bio.base.config.extractor.linearize:extractor',
      ],

      'bob.bio.algorithm': [
        'dummy             = bob.bio.base.test.dummy.algorithm:algorithm', # for test purposes only
        'dummy_metadata    = bob.bio.base.test.dummy.algorithm:algorithm_metadata',
        'distance-euclidean = bob.bio.base.config.algorithm.distance_euclidean:algorithm',
        'distance-cosine   = bob.bio.base.config.algorithm.distance_cosine:algorithm',
        'distance-hamming   = bob.bio.base.config.algorithm.distance_hamming:algorithm',
        'pca               = bob.bio.base.config.algorithm.pca:algorithm',
        'lda               = bob.bio.base.config.algorithm.lda:algorithm',
        'pca+lda           = bob.bio.base.config.algorithm.pca_lda:algorithm',
        'plda              = bob.bio.base.config.algorithm.plda:algorithm',
        'pca+plda          = bob.bio.base.config.algorithm.pca_plda:algorithm',
        'bic               = bob.bio.base.config.algorithm.bic:algorithm',
      ],

      'bob.bio.grid': [
        'local-p4          = bob.bio.base.config.grid.local:grid',
        'local-p8          = bob.bio.base.config.grid.local:grid_p8',
        'local-p16         = bob.bio.base.config.grid.local:grid_p16',
        'grid              = bob.bio.base.config.grid.grid:grid',
        'demanding         = bob.bio.base.config.grid.demanding:grid',
        'gpu               = bob.bio.base.config.grid.gpu:grid',
      ],

      # declare database to bob
      'bob.db': [
        'bio_filelist      = bob.bio.base.database.filelist.driver:Interface',
      ],
      # main entry for bob bio cli
      'bob.cli': [
        'bio               = bob.bio.base.script.bio:bio',
      ],

      # bob bio scripts
      'bob.bio.cli': [
        'annotate          = bob.bio.base.script.annotate:annotate',
      ],

      # annotators
      'bob.bio.annotator': [
        'dummy             = bob.bio.base.test.dummy.annotator:annotator',
      ],
   },

    # Classifiers are important if you plan to distribute this package through
    # PyPI. You can find the complete list of classifiers that are valid and
    # useful here (http://pypi.python.org/pypi?%3Aaction=list_classifiers).
    classifiers = [
      'Framework :: Bob',
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: BSD License',
      'Natural Language :: English',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
