from ..utils.processors import SequentialProcessor, ParallelProcessor
from .Preprocessor import Preprocessor


class SequentialPreprocessor(SequentialProcessor, Preprocessor):
    __doc__ = SequentialProcessor.__doc__

    def __init__(self, processors, **kwargs):
        min_preprocessed_file_size = 1000
        try:
            min_preprocessed_file_size = min(
                (p.min_preprocessed_file_size for p in processors))
        except AttributeError:
            pass

        SequentialProcessor.__init__(self, processors)
        Preprocessor.__init__(
            self, min_preprocessed_file_size=min_preprocessed_file_size,
            **kwargs)

    def __call__(self, data, annotations):
        return super(SequentialPreprocessor, self).__call__(
            data, annotations=annotations)

    def read_data(self, data_file):
        return self.processors[-1].read_data(data_file)

    def write_data(self, data, data_file):
        return self.processors[-1].write_data(data, data_file)


class ParallelPreprocessor(ParallelProcessor, Preprocessor):
    __doc__ = ParallelProcessor.__doc__

    def __init__(self, processors, **kwargs):
        min_preprocessed_file_size = min(p.min_preprocessed_file_size for p in
                                         processors)

        ParallelProcessor.__init__(self, processors)
        Preprocessor.__init__(
            self, min_preprocessed_file_size=min_preprocessed_file_size,
            **kwargs)

    def __call__(self, data, annotations):
        return super(ParallelPreprocessor, self).__call__(
            data, annotations=annotations)


class CallablePreprocessor(Preprocessor):
    """A simple preprocessor that takes a callable and applies that callable to
    the input.

    Attributes
    ----------
    accepts_annotations : bool
        If False, annotations are not passed to the callable.
    callable : object
        Anything that is callable. It will be used as a preprocessor in
        bob.bio.base.
    """

    def __init__(self, callable, accepts_annotations=True, **kwargs):
        super(CallablePreprocessor, self).__init__(
            callable=callable, accepts_annotations=accepts_annotations,
            **kwargs)
        self.callable = callable
        self.accepts_annotations = accepts_annotations

    def __call__(self, data, annotations):
        if self.accepts_annotations:
            return self.callable(data, annotations)
        else:
            return self.callable(data)
