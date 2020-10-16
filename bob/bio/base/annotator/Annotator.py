from sklearn.base import TransformerMixin, BaseEstimator


class Annotator(TransformerMixin, BaseEstimator):
    """Annotator class for all annotators. This class is meant to be used in
    conjunction with the bob bio annotate script or to be used in pipelines.

    Attributes
    ----------
    read_original_data : callable
        A function that loads the samples. The syntax is like
        `bob.bio.base.read_original_data`.
    """

    def transform(self, samples, **kwargs):
        """Annotates a sample and returns annotations in a dictionary.

        Parameters
        ----------
        samples : numpy.ndarray
            The samples that are being annotated.
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
    def __call__(self, samples, **kwargs):
        return self.transform(samples, **kwargs)
