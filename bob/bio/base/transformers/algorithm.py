from sklearn.base import TransformerMixin, BaseEstimator
from bob.bio.base.algorithm import Algorithm
from . import split_X_by_y, _LazyModelLoader
import bob.pipelines as mario


class TransformerAlgorithm(_LazyModelLoader, TransformerMixin, BaseEstimator):
    """Wraps a :any:`bob.bio.base.algorithm.Algorithm` to become a Scikit-learn
    transformer.

    :py:method:`TransformerAlgorithm.fit` maps to :py:method:`bob.bio.base.algorithm.Algoritm.train_projector`

    :py:method:`TransformerAlgorithm.transform` maps :py:method:`bob.bio.base.algorithm.Algoritm.project`

    Example
    -------
    Wrapping an LDA algorithm:

    >>> from bob.bio.base.algorithm import LDA
    >>> transformer = TransformerAlgorithm(LDA(use_pinv=True, pca_subspace_dimension=0.90))


    Attributes
    ----------
    instance : object
        Instance of `bob.bio.base.algorithm.Algorithm`
    model_path : str
        Model path in case ``instance.requires_projector_training`` is True.
    """

    def __init__(
        self, instance, model_path=None, **kwargs,
    ):
        if not isinstance(instance, Algorithm):
            raise ValueError(
                f"{instance} should be an instance of `bob.bio.base.algorithm.Algorithm`"
            )

        if not instance.performs_projection:
            raise ValueError(
                f"{instance} should perform projections to be used here. "
                "``algorithm.performs_projection`` should be True."
            )

        super().__init__(
            model_path=model_path, load_method_name="load_projector", **kwargs
        )
        self.instance = instance
        self.model_path = model_path
        if model_path is None and instance.requires_projector_training:
            raise ValueError(
                f"Algorithm: {instance} requires_projector_training. Hence, "
                "`model_path` cannot be None."
            )

    def transform(self, X):
        self.load()
        return [self.instance.project(feature) for feature in X]

    def fit(self, X, y=None):
        if not self.instance.requires_projector_training:
            return self

        # if the model exists, load it instead
        if self.load():
            return self

        training_data = X
        if self.instance.split_training_features_by_client:
            training_data = split_X_by_y(X, y)

        self.instance.train_projector(training_data, self.model_path)
        return self

    def _more_tags(self):
        return {"requires_fit": self.instance.requires_projector_training}


def _prepare_algorithm_sample_args(algorithm, fit_extra_arguments):

    if (
        fit_extra_arguments is None
        and algorithm.requires_projector_training
        and algorithm.split_training_features_by_client
    ):
        fit_extra_arguments = (("y", "subject"),)

    return fit_extra_arguments


def SampleAlgorithm(algorithm, model_path=None, fit_extra_arguments=None, **kwargs):
    """Wraps a :any:`bob.bio.base.algorithm.Algorithm` to become a sample-based
    transformer.

    Parameters
    ----------
    algorithm : object
        Instance of :any:`bob.bio.base.algorithm.Algorithm`
    model_path : str, optional
        See :any:`TransformerAlgorithm`.
    fit_extra_arguments : None, optional
        See :any:`bob.pipelines.SampleWrapper`.
    **kwargs
        Extra arguments passed to :any:`bob.pipelines.SampleWrapper`.

    Returns
    -------
    object
        The wrapped transformer.
    """
    transformer = TransformerAlgorithm(algorithm, model_path=model_path)

    fit_extra_arguments = _prepare_algorithm_sample_args(algorithm, fit_extra_arguments)

    return mario.wrap(
        ["sample"], transformer, fit_extra_arguments=fit_extra_arguments, **kwargs,
    )


def CheckpointAlgorithm(
    algorithm,
    model_path=None,
    features_dir=None,
    extension=".hdf5",
    load_func=None,
    save_func=None,
    fit_extra_arguments=None,
    **kwargs,
):
    """Wraps a :any:`bob.bio.base.algorithm.Algorithm` to become a sample-based
    and checkpointing transformer.

    Parameters
    ----------
    algorithm : object
        Instance of :any:`bob.bio.base.algorithm.Algorithm`
    model_path : str, optional
        See :any:`TransformerAlgorithm`.
    features_dir : None, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    extension : str, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    load_func : None, optional
        See :any:bob.pipelines.CheckpointWrapper`.
    save_func : None, optional
        See :any:bob.pipelines.CheckpointWrapper`.
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
    transformer = TransformerAlgorithm(algorithm, model_path=model_path)

    fit_extra_arguments = _prepare_algorithm_sample_args(algorithm, fit_extra_arguments)

    return mario.wrap(
        ["sample", "checkpoint"],
        transformer,
        fit_extra_arguments=fit_extra_arguments,
        features_dir=features_dir,
        extension=extension,
        load_func=load_func or algorithm.read_feature,
        save_func=save_func or algorithm.write_feature,
        **kwargs,
    )
