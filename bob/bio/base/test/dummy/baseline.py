from bob.bio.base.baseline import Baseline
import pkg_resources
import os

dummy_dir = pkg_resources.resource_filename('bob.bio.base', 'test/dummy')
baseline = Baseline(name="dummy", 
                         preprocessors={"default": os.path.join(dummy_dir, 'preprocessor.py')},
                         extractor=os.path.join(dummy_dir, 'extractor.py'),
                         algorithm=os.path.join(dummy_dir, 'algorithm.py'))
