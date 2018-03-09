from random import random
from bob.bio.base.annotator import FailSafe, Callable


def moody_annotator(image, **kwargs):
    if random() < 0.5:
        return {
            'topleft': (0, 0),
        }
    else:
        return {
            'topleft': (0, 0),
            'bottomright': image.shape,
        }


def fail_annotator(image, **kwargs):
    return {}


annotator = FailSafe(
    [Callable(fail_annotator),
     Callable(moody_annotator)],
    required_keys=['topleft', 'bottomright'],
)
