from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.preprocessor import Preprocessor
import bob.pipelines as mario


class TransformerPreprocessor(TransformerMixin, BaseEstimator):
    """Wraps a :any:`bob.bio.base.preprocessor.Preprocessor` to become a
    Scikit-learn transformer.

    Attributes
    ----------
    instance : object
        Instance of `bob.bio.base.preprocessor.Preprocessor`
    """

    def __init__(
        self, instance, **kwargs,
    ):
        if not isinstance(instance, Preprocessor):
            raise ValueError(
                f"{instance} should be an instance of `bob.bio.base.preprocessor.Preprocessor`"
            )

        super().__init__(**kwargs)
        self.instance = instance

    def transform(self, X, annotations=None):
        if annotations is None:
            return [self.instance(data) for data in X]
        else:
            return [self.instance(data, annot) for data, annot in zip(X, annotations)]

    def _more_tags(self):
        return {"stateless": True, "requires_fit": False}

    def fit(self, X, y=None, **fit_params):
        return self


def SamplePreprocessor(
    preprocessor, transform_extra_arguments=(("annotations", "annotations"),), **kwargs
):
    """Wraps a :any:`bob.bio.base.preprocessor.Preprocessor` to become a
    sample-based transformer.

    Parameters
    ----------
    preprocessor : object
        Instance of :any:`bob.bio.base.preprocessor.Preprocessor`
    transform_extra_arguments : tuple, optional
        See :any:`bob.pipelines.SampleWrapper`.
    **kwargs
        Extra arguments passed to :any:`bob.pipelines.SampleWrapper`.

    Returns
    -------
    object
        The wrapped transformer.
    """
    transformer = TransformerPreprocessor(preprocessor)
    return mario.wrap(
        ["sample"],
        transformer,
        transform_extra_arguments=transform_extra_arguments,
        **kwargs,
    )


def CheckpointPreprocessor(
    preprocessor,
    features_dir=None,
    extension=".hdf5",
    load_func=None,
    save_func=None,
    transform_extra_arguments=(("annotations", "annotations"),),
    **kwargs,
):
    """Wraps a :any:`bob.bio.base.preprocessor.Preprocessor` to become a
    sample-based and checkpointing transformer.

    Parameters
    ----------
    preprocessor : object
        Instance of :any:`bob.bio.base.preprocessor.Preprocessor`
    features_dir : None, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    extension : str, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    load_func : None, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    save_func : None, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    transform_extra_arguments : tuple, optional
        See :any:bob.pipelines.SampleWrapper`.
    **kwargs
        Extra arguments passed to :any:bob.pipelines.SampleWrapper` and
        :any:bob.pipelines.CheckpointWrapper`.

    Returns
    -------
    object
        The wrapped transformer.
    """
    transformer = TransformerPreprocessor(preprocessor)
    return mario.wrap(
        ["sample", "checkpoint"],
        transformer,
        transform_extra_arguments=transform_extra_arguments,
        features_dir=features_dir,
        extension=extension,
        load_func=load_func or preprocessor.read_data,
        save_func=save_func or preprocessor.write_data,
        **kwargs,
    )
