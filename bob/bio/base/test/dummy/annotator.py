from bob.bio.base.annotator import FailSafe, Callable


def simple_annotator(image, **kwargs):
    return {
        'topleft': (0, 0),
        'bottomright': image.shape,
    }


def fail_annotator(image, **kwargs):
    return {}


annotator = FailSafe(
    [Callable(fail_annotator),
     Callable(simple_annotator)],
    required_keys=['topleft', 'bottomright'],
)
