from bob.extension.processors import SequentialProcessor, ParallelProcessor
from .Preprocessor import Preprocessor


class SequentialPreprocessor(SequentialProcessor, Preprocessor):
    """A helper class which takes several preprocessors and applies them one by
    one sequentially.

    Attributes
    ----------
    processors : list
        A list of preprocessors to apply.

    Examples
    --------
    You can use this class to apply a chain of preprocessors on your data. For
    example:

    >>> import numpy as np
    >>> from functools import  partial
    >>> from bob.bio.base.preprocessor import SequentialPreprocessor, CallablePreprocessor
    >>> raw_data = np.array([[1, 2, 3], [1, 2, 3]])
    >>> seq_preprocessor = SequentialPreprocessor(
    ...     [CallablePreprocessor(f, accepts_annotations=False) for f in
    ...      [np.cast['float64'], lambda x: x / 2, partial(np.mean, axis=1)]])
    >>> seq_preprocessor(raw_data)
    array([ 1.,  1.])
    >>> np.all(seq_preprocessor(raw_data) ==
    ...        np.mean(np.cast['float64'](raw_data) / 2, axis=1))
    True
    """

    def __init__(self, processors, read_original_data=None, **kwargs):
        min_preprocessed_file_size = min(
            (p.min_preprocessed_file_size for p in processors))
        if read_original_data is None:
            read_original_data = processors[0].read_original_data
        super(SequentialPreprocessor, self).__init__(
            processors=processors,
            min_preprocessed_file_size=min_preprocessed_file_size,
            read_original_data=read_original_data,
            **kwargs)

    def __call__(self, data, annotations=None):
        return super(SequentialPreprocessor, self).__call__(
            data, annotations=annotations)

    def read_data(self, data_file):
        return self.processors[-1].read_data(data_file)

    def write_data(self, data, data_file):
        self.processors[-1].write_data(data, data_file)


class ParallelPreprocessor(ParallelProcessor, Preprocessor):
    """A helper class which takes several preprocessors and applies them on
    each processor separately and yields their outputs one by one.

    Attributes
    ----------
    processors : list
        A list of preprocessors to apply.

    Examples
    --------
    You can use this class to apply several preprocessors on your data and get
    all the results back. For example:

    >>> import numpy as np
    >>> from functools import  partial
    >>> from bob.bio.base.preprocessor import ParallelPreprocessor, CallablePreprocessor
    >>> raw_data = np.array([[1, 2, 3], [1, 2, 3]])
    >>> parallel_preprocessor = ParallelPreprocessor(
    ...     [CallablePreprocessor(f, accepts_annotations=False) for f in
    ...      [np.cast['float64'], lambda x: x / 2.0]])
    >>> list(parallel_preprocessor(raw_data))
    [array([[ 1.,  2.,  3.],
           [ 1.,  2.,  3.]]), array([[ 0.5,  1. ,  1.5],
           [ 0.5,  1. ,  1.5]])]

    The data may be further processed using a :any:`SequentialPreprocessor`:

    >>> from bob.bio.base.preprocessor import SequentialPreprocessor
    >>> total_preprocessor = SequentialPreprocessor(
    ...     [parallel_preprocessor, CallablePreprocessor(list, False),
    ...      CallablePreprocessor(partial(np.concatenate, axis=1), False)])
    >>> total_preprocessor(raw_data)
    array([[ 1. ,  2. ,  3. ,  0.5,  1. ,  1.5],
           [ 1. ,  2. ,  3. ,  0.5,  1. ,  1.5]])
    """

    def __init__(self, processors, **kwargs):
        min_preprocessed_file_size = min(p.min_preprocessed_file_size for p in
                                         processors)

        super(ParallelPreprocessor, self).__init__(
            processors=processors,
            min_preprocessed_file_size=min_preprocessed_file_size,
            **kwargs)

    def __call__(self, data, annotations=None):
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
    read_data : object
        A callable object with the signature of
        ``data = read_data(data_file)``. If not provided, the default
        implementation handles numpy arrays.
    write_data : object
        A callable object with the signature of
        ``write_data(data, data_file)``. If not provided, the default
        implementation handles numpy arrays.

    Examples
    --------
    You can take any function like ``numpy.cast['float32']`` to cast your data
    to float32 for example. This is useful when you want to stack several
    preprocessors using the :any:`SequentialPreprocessor` and
    :any:`ParallelPreprocessor` classes.
    """

    def __init__(self, callable, accepts_annotations=True, write_data=None,
                 read_data=None, **kwargs):
        super(CallablePreprocessor, self).__init__(
            callable=callable, accepts_annotations=accepts_annotations,
            **kwargs)
        self.callable = callable
        self.accepts_annotations = accepts_annotations
        if write_data is not None:
            self.write_data = write_data
        if read_data is not None:
            self.read_data = read_data

    def __call__(self, data, annotations=None):
        if self.accepts_annotations:
            return self.callable(data, annotations)
        else:
            return self.callable(data)
