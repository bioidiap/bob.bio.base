from bob.pipelines import CheckpointWrapper, SampleSet
from bob.pipelines.wrappers import _frmt

from os.path import dirname, isfile, expanduser, join
from os import makedirs
import logging
import json

logger = logging.getLogger(__name__)

class SaveAnnotationsWrapper(CheckpointWrapper):
    """
    A specialization of bob.pipelines.CheckpointWrapper that saves annotations.

    Saves :py:attr:`~bob.pipelines.Sample.annotations` to the disk instead of
    :py:attr:`~bob.pipelines.Sample.data` (default in
    :py:class:`~bob.pipelines.CheckpointWrapper`).

    The annotations of each sample will be "dumped" with json in a file
    corresponding to the one in the original dataset (following the same path
    structure, ie. using the :py:attr:`~bob.pipelines.Sample.key` attribute of
    each sample).

    Parameters
    ----------

    estimator: Annotator Transformer
        Transformer that places samples annotations in
        :py:attr:`~bob.pipelines.Sample.annotations`.

    annotations_dir: str
        The root path where the annotations will be saved.

    extension: str
        The extension of the annotations files [default: ``.json``].

    save_func: function
        The function used to save each sample [default: :py:func:`json.dump`].

    overwrite: bool
        when ``True``, will overwrite any existing files. Otherwise, will skip
        samples when an annotation file with the same ``key`` exists.
    """

    def __init__(
        self,
        estimator,
        annotations_dir,
        extension=".json",
        save_func=None,
        overwrite=False,
        **kwargs,
    ):
        save_func = save_func or self._save_json
        super(SaveAnnotationsWrapper, self).__init__(
            estimator=estimator,
            features_dir=annotations_dir,
            extension=extension,
            save_func=save_func,
            **kwargs,
        )
        self.overwrite = overwrite

    def save(self, sample):
        """
        Saves one sample's annotations to a file on disk.

        Overrides :py:meth:`bob.pipelines.CheckpointWrapper.save`

        Parameters
        ----------

        sample: :py:class:`~bob.pipelines.Sample`
            One sample containing an :py:attr:`~bob.pipelinessSample.annotations`
            attribute.
        """
        path = self.make_path(sample)
        makedirs(dirname(path), exist_ok=True)
        try:
            self.save_func(sample.annotations, path)
        except Exception as e:
            raise RuntimeError(
                f"Could not save annotations of {sample}\n"
                f"(annotations are: {sample.annotations})\n"
                f"during {self}.save"
            ) from e

    def _checkpoint_transform(self, samples, method_name):
        """
        Checks if a transform needs to be saved to the disk.

        Overrides :py:meth:`bob.pipelines.CheckpointWrapper._checkpoint_transform`
        """
        # Transform either samples or samplesets
        method = getattr(self.estimator, method_name)
        logger.debug(f"{_frmt(self)}.{method_name}")

        # if features_dir is None, just transform all samples at once
        if self.features_dir is None:
            return method(samples)

        def _transform_samples(samples):
            paths = [self.make_path(s) for s in samples]
            should_compute_list = [
                p is None or not isfile(p) or self.overwrite
                for p in paths
            ]
            skipped_count = len([s for s in should_compute_list if s==False])
            if skipped_count != 0:
                logger.info(f"Skipping {skipped_count} already existing files.")
            # call method on non-checkpointed samples
            non_existing_samples = [
                s
                for s, should_compute in zip(samples, should_compute_list)
                if should_compute
            ]
            # non_existing_samples could be empty
            computed_features = []
            if non_existing_samples:
                computed_features = method(non_existing_samples)
            # return computed features and checkpointed features
            features, com_feat_index = [], 0
            for s, p, should_compute in zip(samples, paths, should_compute_list):
                if should_compute:
                    feat = computed_features[com_feat_index]
                    com_feat_index += 1
                    # save the computed feature
                    if p is not None:
                        self.save(feat)
                        feat = self.load(s, p)
                    features.append(feat)
                else:
                    features.append(self.load(s, p))
            return features

        if isinstance(samples[0], SampleSet):
            return [SampleSet(_transform_samples(s.samples), parent=s) for s in samples]
        else:
            return _transform_samples(samples)

    def _save_json(self, annot, path):
        """
        Saves the annotations in json format in the file ``path``.

        This is the default ``save_func`` if it is not passed as parameters of
        :py:class:`~bob.bio.base.annotator.SaveAnnotationsWrapper`.

        Parameters
        ----------

        annot: dict
            Any dictionary (containing annotations for example).

        path: str
            A filename pointing in an existing directory.
        """
        logger.debug(f"Writing annotations '{annot}' to file '{path}'.")
        with open(path, "w") as f:
            json.dump(annot, f, indent=1, allow_nan=False)