import bob.bio.base.annotator

import logging
import numpy
import time

logger = logging.getLogger(__name__)

class DummyAnnotator(bob.bio.base.annotator.Annotator):

    def __init__(self, **kwargs):
        super(DummyAnnotator, self).__init__(**kwargs)

    def transform(self, sample, **kwargs):
        for s in sample:
            logger.debug(f"Annotating sample: {s.key}")
            s.annotations = {
                "time": time.localtime(),
                "rand": list(numpy.random.uniform(0,1,2))
                }
            time.sleep(0.1)
        return sample


annotator = DummyAnnotator()