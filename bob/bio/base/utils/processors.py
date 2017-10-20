class SequentialProcessor(object):
    """A helper class which takes several processors and applies them one by
    one sequentially

    Attributes
    ----------
    processors : list
        A list of processors to apply.
    """

    def __init__(self, processors, **kwargs):
        super(SequentialProcessor, self).__init__()
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
    """

    def __init__(self, processors, **kwargs):
        super(ParallelProcessor, self).__init__()
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
