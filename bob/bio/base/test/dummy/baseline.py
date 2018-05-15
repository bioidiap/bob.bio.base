from bob.bio.base.baseline import Baseline
import pkg_resources
import os


dummy_dir = pkg_resources.resource_filename('bob.bio.base', 'test/dummy')
class DummyBaseline(Baseline):

    def __init__(self, **kwargs):
        super(DummyBaseline, self).__init__(**kwargs)

baseline = DummyBaseline(name="dummy", 
                         preprocessors={"default": os.path.join(dummy_dir, 'preprocessor.py')},
                         extractor=os.path.join(dummy_dir, 'extractor.py'),
                         algorithm=os.path.join(dummy_dir, 'algorithm.py'))
