from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.extractor import Extractor
from bob.bio.base.utils import is_argument_available
from . import split_X_by_y, _LazyModelLoader
import bob.pipelines as mario


class TransformerExtractor(_LazyModelLoader, TransformerMixin, BaseEstimator):
    """Wraps a :any:`bob.bio.base.extractor.Extractor` to become a Scikit-learn
    transformer.

    Attributes
    ----------
    instance : object
        Instance of `bob.bio.base.extractor.Extractor`
    requires_metadata : bool
        Whether the extractor accepts metadata in its __call__ method or not.
    model_path : str
        Model path in case ``instance.requires_training`` is True.
    """

    def __init__(
        self, instance, model_path=None, **kwargs,
    ):
        if not isinstance(instance, Extractor):
            raise ValueError(
                f"{instance} should be an instance of `bob.bio.base.extractor.Extractor`"
            )
        super().__init__(model_path=model_path, **kwargs)
        self.instance = instance
        self.requires_metadata = False
        if is_argument_available("metadata", instance.__call__):
            self.requires_metadata = True
        if instance.requires_training and model_path is None:
            raise ValueError(
                f"The extractor {instance} requires training and model_path cannot be None."
            )

    def transform(self, X, metadata=None):
        self.load()
        if self.requires_metadata:
            return [self.instance(data, metadata=m) for data, m in zip(X, metadata)]
        else:
            return [self.instance(data) for data in X]

    def fit(self, X, y=None):
        if not self.instance.requires_training:
            return self

        # if the model exists, load it instead
        if self.load():
            return self

        training_data = X
        if self.instance.split_training_data_by_client:
            training_data = split_X_by_y(X, y)

        self.instance.train(training_data, self.model_path)

        return self

    def _more_tags(self):
        return {
            "requires_fit": self.instance.requires_training,
            "stateless": not self.instance.requires_training,
        }


def _prepare_extractor_sample_args(
    extractor, transform_extra_arguments, fit_extra_arguments
):

    if transform_extra_arguments is None and is_argument_available(
        "metadata", extractor.__call__
    ):
        transform_extra_arguments = (("metadata", "metadata"),)

    if (
        fit_extra_arguments is None
        and extractor.requires_training
        and extractor.split_training_data_by_client
    ):
        fit_extra_arguments = (("y", "subject"),)

    return transform_extra_arguments, fit_extra_arguments


def SampleExtractor(
    extractor,
    model_path=None,
    transform_extra_arguments=None,
    fit_extra_arguments=None,
    **kwargs,
):
    """Wraps a :any:`bob.bio.base.extractor.Extractor` to become a sample-based
    transformer.

    Parameters
    ----------
    extractor : object
        Instance of :any:`bob.bio.base.extractor.Extractor`
    model_path : str, optional
        See :any:`TransformerExtractor`.
    transform_extra_arguments : tuple, optional
        See :any:`bob.pipelines.SampleWrapper`.
    fit_extra_arguments : None, optional
        See :any:`bob.pipelines.SampleWrapper`.
    **kwargs
        Extra arguments passed to :any:`bob.pipelines.SampleWrapper`.

    Returns
    -------
    object
        The wrapped transformer.
    """
    transformer = TransformerExtractor(extractor, model_path=model_path)

    transform_extra_arguments, fit_extra_arguments = _prepare_extractor_sample_args(
        extractor, transform_extra_arguments, fit_extra_arguments
    )

    return mario.wrap(
        ["sample"],
        transformer,
        transform_extra_arguments=transform_extra_arguments,
        fit_extra_arguments=fit_extra_arguments,
        **kwargs,
    )


def CheckpointExtractor(
    extractor,
    model_path=None,
    features_dir=None,
    extension=".hdf5",
    load_func=None,
    save_func=None,
    transform_extra_arguments=None,
    fit_extra_arguments=None,
    **kwargs,
):
    """Wraps a :any:`bob.bio.base.extractor.Extractor` to become a sample-based
    and checkpointing transformer.

    Parameters
    ----------
    extractor : object
        Instance of :any:`bob.bio.base.extractor.Extractor`
    model_path : str, optional
        See :any:`TransformerExtractor`.
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
    fit_extra_arguments : None, optional
        See :any:`bob.pipelines.SampleWrapper`.
    **kwargs
        Extra arguments passed to :any:bob.pipelines.SampleWrapper` and
        :any:bob.pipelines.CheckpointWrapper`.

    Returns
    -------
    object
        The wrapped transformer.
    """
    transformer = TransformerExtractor(extractor, model_path=model_path)

    transform_extra_arguments, fit_extra_arguments = _prepare_extractor_sample_args(
        extractor, transform_extra_arguments, fit_extra_arguments
    )

    return mario.wrap(
        ["sample", "checkpoint"],
        transformer,
        transform_extra_arguments=transform_extra_arguments,
        fit_extra_arguments=fit_extra_arguments,
        features_dir=features_dir,
        extension=extension,
        load_func=load_func or extractor.read_feature,
        save_func=save_func or extractor.write_feature,
        **kwargs,
    )
