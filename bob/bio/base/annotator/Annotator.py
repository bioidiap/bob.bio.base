from bob.bio.base import read_original_data as base_read


class Annotator(object):
    """Annotator class for all annotators. This class is meant to be used in
    conjunction with the bob bio annotate script.

    Attributes
    ----------
    read_original_data : callable
        A function that loads the samples. The syntax is like
        :any:`bob.bio.base.read_original_data`.
    """

    def __init__(self, read_original_data=None, **kwargs):
        super(Annotator, self).__init__(**kwargs)
        self.read_original_data = read_original_data or base_read

    def annotate(self, sample, **kwargs):
        """Annotates a sample and returns annotations in a dictionary.

        Parameters
        ----------
        sample : numpy.ndarray
            The sample that is being annotated.
        **kwargs
            The extra arguments that may be passed.

        Returns
        -------
        dict
            A dictionary containing the annotations of the biometric sample. If
            the program fails to annotate the sample, it should return an empty
            dictionary.
        """
        raise NotImplementedError

    # Alias call to annotate
    def __call__(self, sample, **kwargs):
        return self.annotate(sample, **kwargs)
