from . import Annotator


class Callable(Annotator):
    """A class that wraps a callable object that annotates a sample into a
    bob.bio.annotator object.

    Attributes
    ----------
    callable
        A callable with the following signature:
        ``annotations = callable(sample, **kwargs)`` that takes numpy array and
        returns annotations in dictionary format for that biometric sample.
        Please see :any:`Annotator` for more information.
    """

    def __init__(self, callable, **kwargs):
        super(Callable, self).__init__(**kwargs)
        self.callable = callable

    def transform(self, sample, **kwargs):
        return self.callable(sample, **kwargs)
