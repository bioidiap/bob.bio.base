from random import random
from bob.bio.base.annotator import FailSafe, Annotator


class SimpleAnnotator(Annotator):
    def transform(self, samples, **kwargs):
        for sample in samples:
            sample.annotations = {
                'topleft': (0, 0),
                'bottomright': sample.data.shape,
            }
        return samples


class MoodyAnnotator(Annotator):
    def transform(self, samples, **kwargs):
        for sample in samples:
            sample.annotations = {'topleft': (0,0)}
            if random() > 0.5:
                sample.annotations['bottomright'] = sample.data.shape
        return samples


class FailAnnotator(Annotator):
    def transform(self, samples, **kwargs):
        return {}


annotator = FailSafe(
    [FailAnnotator(),
     SimpleAnnotator()],
    required_keys=['topleft', 'bottomright'],
)
