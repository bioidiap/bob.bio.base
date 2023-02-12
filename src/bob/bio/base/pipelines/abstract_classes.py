#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import logging
import os

from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Optional, Union

import numpy as np

from sklearn.base import BaseEstimator

from bob.pipelines import Sample, SampleBatch, SampleSet
from bob.pipelines.wrappers import _frmt

logger = logging.getLogger(__name__)


def reduce_scores(
    scores: np.ndarray,
    axis: int,
    fn: Union[str, Callable[[np.ndarray, int], np.ndarray]] = "max",
):
    """
    Reduce scores using a function.

    Parameters:
    -----------
    scores
        Scores to reduce.

    fn
        Function to use for reduction. You can also provide a string like
        ``max`` to use the corresponding function from numpy. Some possible
        values are: ``max``, ``min``, ``mean``, ``median``, ``sum``.

    Returns:
    --------
    Reduced scores.
    """
    if isinstance(fn, str):
        fn = getattr(np, fn)
    return fn(scores, axis=axis)


def _data_valid(data: Any) -> bool:
    """Check if data is valid.

    Parameters:
    -----------
    data
        Data to check.

    Returns:
    --------
    True if data is valid, False otherwise.
    """
    if data is None:
        return False
    if isinstance(data, np.ndarray):
        return data.size > 0
    # we also have to check for [[]]
    if isinstance(data, list) and len(data) > 0:
        if isinstance(data[0], (list, tuple)):
            return len(data[0]) > 0
    return bool(data)


class BioAlgorithm(BaseEstimator, metaclass=ABCMeta):
    """Describes a base biometric comparator for the PipelineSimple
    :ref:`bob.bio.base.biometric_algorithm`.

    A biometric algorithm converts each SampleSet (which is a list of
    samples/features) into a single template. Template creation is done for both
    enroll and probe samples but the format of the templates can be different
    between enrollment and probe samples. After the creation of the templates,
    the algorithm computes one similarity score for comparison of an enroll
    template with a probe template.

    Examples
    --------
    >>> import numpy as np
    >>> from bob.bio.base.pipelines import BioAlgorithm
    >>> class MyAlgorithm(BioAlgorithm):
    ...
    ...     def create_templates(self, list_of_feature_sets, enroll):
    ...         # you cannot call np.mean(list_of_feature_sets, axis=1) because the
    ...         # number of features in each feature set may vary.
    ...         return [np.mean(feature_set, axis=0) for feature_set in list_of_feature_sets]
    ...
    ...     def compare(self, enroll_templates, probe_templates):
    ...         scores = []
    ...         for enroll_template in enroll_templates:
    ...             scores.append([])
    ...             for probe_template in probe_templates:
    ...                 similarity = 1 / np.linalg.norm(model - probe)
    ...                 scores[-1].append(similarity)
    ...         scores = np.array(scores, dtype=float)
    ...         return scores
    """

    def __init__(
        self,
        probes_score_fusion: Union[
            str, Callable[[list[np.ndarray], int], np.ndarray]
        ] = "max",
        enrolls_score_fusion: Union[
            str, Callable[[list[np.ndarray], int], np.ndarray]
        ] = "max",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.probes_score_fusion = probes_score_fusion
        self.enrolls_score_fusion = enrolls_score_fusion

    def fuse_probe_scores(self, scores, axis):
        return reduce_scores(scores, axis, self.probes_score_fusion)

    def fuse_enroll_scores(self, scores, axis):
        return reduce_scores(scores, axis, self.enrolls_score_fusion)

    @abstractmethod
    def create_templates(
        self, list_of_feature_sets: list[Any], enroll: bool
    ) -> list[Sample]:
        """Creates enroll or probe templates from multiple sets of features.

        The enroll template format can be different from the probe templates.

        Parameters
        ----------
        list_of_feature_sets
            A list of list of features with the shape of Nx?xD. N templates
            should be computed. Note that you cannot call
            np.array(list_of_feature_sets) because the number of features per
            set can be different depending on the database.
        enroll
            If True, the features are for enrollment. If False, the features are
            for probe.

        Returns
        -------
        templates
            A list of templates which has the same length as
            ``list_of_feature_sets``.
        """
        pass

    @abstractmethod
    def compare(
        self, enroll_templates: list[Sample], probe_templates: list[Sample]
    ) -> np.ndarray:
        """Computes the similarity score between all enrollment and probe templates.

        Parameters
        ----------
        enroll_templates
            A list (length N) of enrollment templates.

        probe_templates
            A list (length M) of probe templates.

        Returns
        -------
        scores
            A matrix of shape (N, M) containing the similarity scores.
        """
        pass

    def create_templates_from_samplesets(
        self, list_of_samplesets: list[SampleSet], enroll: bool
    ) -> list[Sample]:
        """Creates enroll or probe templates from multiple SampleSets.

        Parameters
        ----------
        list_of_samplesets
            A list (length N) of SampleSets.

        enroll
            If True, the SampleSets are for enrollment. If False, the SampleSets
            are for probe.

        Returns
        -------
        templates
            A list of Samples which has the same length as ``list_of_samplesets``.
            Each Sample contains a template.
        """
        logger.debug(
            f"{_frmt(self)}.create_templates_from_samplesets(... enroll={enroll})"
        )
        # create templates from .data attribute of samples inside sample_sets
        list_of_feature_sets = []
        for sampleset in list_of_samplesets:
            data = [s.data for s in sampleset.samples]
            valid_data = [d for d in data if d is not None]
            if len(data) != len(valid_data):
                logger.warning(
                    f"Removed {len(data)-len(valid_data)} invalid enrollment samples."
                )
            if not valid_data and enroll:
                # we do not support failure to enroll cases currently
                raise NotImplementedError(
                    f"None of the enrollment samples were valid for {sampleset}."
                )
            list_of_feature_sets.append(valid_data)

        templates = self.create_templates(list_of_feature_sets, enroll)
        expected_size = len(list_of_samplesets)
        assert len(templates) == expected_size, (
            "The number of (%s) templates (%d) created by the algorithm does not match "
            "the number of sample sets (%d)"
            % (
                "enroll" if enroll else "probe",
                len(templates),
                expected_size,
            )
        )
        # return a list of Samples (one per template)
        templates = [
            Sample(t, parent=sampleset)
            for t, sampleset in zip(templates, list_of_samplesets)
        ]
        return templates

    def score_sample_templates(
        self,
        probe_samples: list[Sample],
        enroll_samples: list[Sample],
        score_all_vs_all: bool,
    ) -> list[SampleSet]:
        """Computes the similarity score between all probe and enroll templates.

        Parameters
        ----------
        probe_samples
            A list (length N) of Samples containing probe templates.

        enroll_samples
            A list (length M) of Samples containing enroll templates.

        score_all_vs_all
            If True, the similarity scores between all probe and enroll templates
            are computed. If False, the similarity scores between the probes and
            their associated enroll templates are computed.

        Returns
        -------
        score_samplesets
            A list of N SampleSets each containing a list of M score Samples if score_all_vs_all
            is True. Otherwise, a list of N SampleSets each containing a list of <=M score Samples
            depending on the database.
        """
        logger.debug(
            f"{_frmt(self)}.score_sample_templates(... score_all_vs_all={score_all_vs_all})"
        )
        # Returns a list of SampleSets where a Sampleset for each probe
        # SampleSet where each Sample inside the SampleSets contains the score
        # for one enroll SampleSet
        score_samplesets = []
        if score_all_vs_all:
            probe_data = [s.data for s in probe_samples]
            valid_probe_indices = [
                i for i, d in enumerate(probe_data) if _data_valid(d)
            ]
            valid_probe_data = [probe_data[i] for i in valid_probe_indices]
            scores = self.compare(SampleBatch(enroll_samples), valid_probe_data)
            scores = np.asarray(scores, dtype=float)

            if len(valid_probe_indices) != len(probe_data):
                # inject None scores for invalid probe samples
                scores: list = scores.T.tolist()
                for i in range(len(probe_data)):
                    if i not in valid_probe_indices:
                        scores.insert(i, [None] * len(enroll_samples))
                # transpose back to original shape
                scores = np.array(scores, dtype=float).T

            expected_shape = (len(enroll_samples), len(probe_samples))
            assert scores.shape == expected_shape, (
                "The shape of the similarity scores (%s) does not match the expected shape (%s)"
                % (scores.shape, expected_shape)
            )
            for j, probe in enumerate(probe_samples):
                samples = []
                for i, enroll in enumerate(enroll_samples):
                    samples.append(Sample(scores[i, j], parent=enroll))
                score_samplesets.append(SampleSet(samples, parent=probe))
        else:
            for probe in probe_samples:
                references = [str(ref) for ref in probe.references]
                # get the indices of references for enroll samplesets
                indices = [
                    i
                    for i, enroll in enumerate(enroll_samples)
                    if str(enroll.template_id) in references
                ]
                if not indices:
                    raise ValueError(
                        f"No enroll sampleset found for probe {probe} and its required references {references}. "
                        "Did you mean to set score_all_vs_all=True?"
                    )
                if not _data_valid(probe.data):
                    scores = [[None]] * len(indices)
                else:
                    scores = self.compare(
                        SampleBatch([enroll_samples[i] for i in indices]),
                        SampleBatch([probe]),
                    )
                scores = np.asarray(scores, dtype=float)
                expected_shape = (len(indices), 1)
                assert scores.shape == expected_shape, (
                    "The shape of the similarity scores (%s) does not match the expected shape (%s)"
                    % (scores.shape, expected_shape)
                )
                samples = []
                for i, j in enumerate(indices):
                    samples.append(
                        Sample(scores[i, 0], parent=enroll_samples[j])
                    )
                score_samplesets.append(SampleSet(samples, parent=probe))

        return score_samplesets


class Database(metaclass=ABCMeta):
    """Base class for PipelineSimple databases"""

    def __init__(
        self,
        protocol: Optional[str] = None,
        score_all_vs_all: bool = False,
        annotation_type: Optional[str] = None,
        fixed_positions: Optional[str] = None,
        memory_demanding: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        protocol
            Name of the database protocol to use.
        score_all_vs_all
            Wether to allow scoring of all the probes against all the references, or to
            provide a list ``references`` provided with each probes to indicate against
            which references it needs to be compared.
        annotation_type
            The type of annotation passed to the annotation loading function.
        fixed_positions
            The constant eyes positions passed to the annotation loading function.
            TODO why keep this face-related name here? Which one is it, too (position
            when annotations are missing, or ending position in the result image)?
            --> move this when the FaceCrop annotator is correctly implemented.
        memory_demanding
            Flag to indicate that this should not be loaded locally.
            TODO Where is it used?
        """
        super().__init__(**kwargs)
        if not hasattr(self, "protocol"):
            self.protocol = protocol
        self.score_all_vs_all = score_all_vs_all
        self.annotation_type = annotation_type
        self.fixed_positions = fixed_positions
        self.memory_demanding = memory_demanding

    def __str__(self) -> str:
        args = ", ".join(
            [
                "{}={}".format(k, v)
                for k, v in self.__dict__.items()
                if not k.startswith("_")
            ]
        )
        return f"{self.__class__.__name__}({args})"

    @abstractmethod
    def background_model_samples(self) -> list[Sample]:
        """Returns :any:`Sample`\ s to train a background model


        Returns
        -------
        samples
            List of samples for background model training.

        """  # noqa: W605
        pass

    @abstractmethod
    def references(self, group: str = "dev") -> list[SampleSet]:
        """Returns references to enroll biometric references


        Parameters
        ----------
        group
            Limits samples to this group


        Returns
        -------
        references
            List of samples for the creation of biometric references.

        """
        pass

    @abstractmethod
    def probes(self, group: str = "dev") -> list[SampleSet]:
        """Returns probes to score against enrolled biometric references


        Parameters
        ----------
        group
            Limits samples to this group


        Returns
        -------
        probes
            List of samples for the creation of biometric probes.

        """
        pass

    @abstractmethod
    def all_samples(self, groups: Optional[str] = None) -> list[Sample]:
        """Returns all the samples of the dataset

        Parameters
        ----------
        groups
            List of groups to consider (like 'dev' or 'eval'). If `None`, will
            return samples from all the groups.

        Returns
        -------
        samples
            List of all the samples of the dataset.
        """
        pass

    @abstractmethod
    def groups(self) -> list[str]:
        """Returns all the possible groups for the current protocol."""
        pass

    @abstractmethod
    def protocols(self) -> list[str]:
        """Returns all the possible protocols of the database."""
        pass

    def template_ids(self, group: str) -> list[Any]:
        """Returns the ``template_id`` attribute of each reference."""
        return [s.template_id for s in self.references(group=group)]


class ScoreWriter(metaclass=ABCMeta):
    """
    Defines base methods to read, write scores and concatenate scores
    for :any:`bob.bio.base.pipelines.BioAlgorithm`
    """

    def __init__(self, path, extension=".txt", **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.extension = extension

    @abstractmethod
    def write(self, sampleset, path):
        pass

    def post_process(self, score_paths, filename):
        def _post_process(score_paths, filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                for path in score_paths:
                    with open(path) as f2:
                        f.writelines(f2.readlines())
            return filename

        import dask
        import dask.bag

        if isinstance(score_paths, dask.bag.Bag):
            all_paths = dask.delayed(list)(score_paths)
            return dask.delayed(_post_process)(all_paths, filename)
        return _post_process(score_paths, filename)
