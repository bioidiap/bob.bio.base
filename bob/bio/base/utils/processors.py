class SequentialProcessor(object):
    """A helper class which takes several processors and applies them one by
    one sequentially.

    Attributes
    ----------
    processors : list
        A list of processors to apply.

    Examples
    --------
    You can use this class to apply a chain of processes on your data. For
    example:

    >>> import numpy as np
    >>> from functools import  partial
    >>> from bob.bio.base.utils.processors import SequentialProcessor
    >>> raw_data = np.array([[1, 2, 3], [1, 2, 3]])
    >>> seq_processor = SequentialProcessor(
    ...     [np.cast['float64'], lambda x: x / 2, partial(np.mean, axis=1)])
    >>> seq_processor(raw_data)
    array([ 1.,  1.])
    >>> np.all(seq_processor(raw_data) ==
    ...        np.mean(np.cast['float64'](raw_data) / 2, axis=1))
    True
    """

    def __init__(self, processors, **kwargs):
        super(SequentialProcessor, self).__init__(**kwargs)
        self.processors = processors

    def __call__(self, data, **kwargs):
        """Applies the processors on the data sequentially. The output of the
        first one goes as input to the next one.

        Parameters
        ----------
        data : object
            The data that needs to be processed.
        **kwargs
            Any kwargs are passed to the processors.

        Returns
        -------
        object
            The processed data.
        """
        for processor in self.processors:
            data = processor(data, **kwargs)
        return data


class ParallelProcessor(object):
    """A helper class which takes several processors and applies them on each
    processor separately and yields their outputs one by one.

    Attributes
    ----------
    processors : list
        A list of processors to apply.

    Examples
    --------
    You can use this class to apply several processes on your data and get all
    the results back. For example:

    >>> import numpy as np
    >>> from functools import  partial
    >>> from bob.bio.base.utils.processors import ParallelProcessor
    >>> raw_data = np.array([[1, 2, 3], [1, 2, 3]])
    >>> parallel_processor = ParallelProcessor(
    ...     [np.cast['float64'], lambda x: x / 2.0])
    >>> list(parallel_processor(raw_data))
    [array([[ 1.,  2.,  3.],
           [ 1.,  2.,  3.]]), array([[ 0.5,  1. ,  1.5],
           [ 0.5,  1. ,  1.5]])]

    The data may be further processed using a :any:`SequentialProcessor`:

    >>> from bob.bio.base.utils.processors import SequentialProcessor
    >>> total_processor = SequentialProcessor(
    ...     [parallel_processor, list, partial(np.concatenate, axis=1)])
    >>> total_processor(raw_data)
    array([[ 1. ,  2. ,  3. ,  0.5,  1. ,  1.5],
           [ 1. ,  2. ,  3. ,  0.5,  1. ,  1.5]])
    """

    def __init__(self, processors, **kwargs):
        super(ParallelProcessor, self).__init__(**kwargs)
        self.processors = processors

    def __call__(self, data, **kwargs):
        """Applies the processors on the data independently and outputs a
        generator of their outputs.

        Parameters
        ----------
        data : object
            The data that needs to be processed.
        **kwargs
            Any kwargs are passed to the processors.

        Yields
        ------
        object
            The processed data from processors one by one.
        """
        for processor in self.processors:
            yield processor(data, **kwargs)
