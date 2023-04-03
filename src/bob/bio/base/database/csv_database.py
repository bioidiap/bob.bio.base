import csv
import functools
import logging
import os

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TextIO

import sklearn.pipeline

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.base.pipelines.abstract_classes import Database
from bob.pipelines import (
    DelayedSample,
    FileListDatabase,
    Sample,
    SampleSet,
    check_parameters_for_validity,
)
from bob.pipelines.dataset import open_definition_file

from ..utils.annotations import read_annotation_file

logger = logging.getLogger(__name__)


def _sample_sets_to_samples(sample_sets):
    return functools.reduce(
        lambda x, y: x + y, (s.samples for s in sample_sets), []
    )


def _add_key(samples: list[Sample]) -> list[Sample]:
    """Adds a ``key`` attribute to all samples if ``key`` is not present

    Will use ``path`` to create a unique ``key``.
    Note that this won't create unique keys if you have multiple times the same path in
    different samples. This will be problematic, as key are expected to be unique.
    """

    out = []
    for sample in samples:
        if isinstance(sample, SampleSet):
            out.append(SampleSet(samples=_add_key(sample), parent=sample))
            continue
        if not hasattr(sample, "key"):
            if hasattr(sample, "path"):
                sample.key = sample.path
            else:
                raise ValueError(
                    f"Sample has no 'key' and no 'path' to infer it. {sample=}"
                )
        out.append(sample)
    return out


def validate_bio_samples(samples):
    """Validates Samples or SampleSets for backwards compatibility reasons.

    This will add a ``key`` attribute (if not already present) to each sample, copied
    from the path.
    """

    for sample in samples:
        if isinstance(sample, SampleSet):
            validate_bio_samples(sample.samples)
            if not hasattr(sample, "template_id"):
                raise ValueError(
                    f"SampleSet must have a template_id attribute, got {sample}"
                )
            if not hasattr(sample, "subject_id"):
                raise ValueError(
                    f"SampleSet must have a subject_id attribute, got {sample}"
                )
            continue

        if not hasattr(sample, "key"):
            if hasattr(sample, "path"):
                sample.key = sample.path
            else:
                raise ValueError(
                    f"Sample must have a key or a path attribute, got {sample}"
                )

        if not hasattr(sample, "subject_id"):
            raise ValueError(
                f"Sample must have a subject_id attribute, got {sample}"
            )


class CSVDatabase(FileListDatabase, Database):
    """A csv file database.

    The database protocol files should provide the following files:


    .. code-block:: text

        dataset_protocols_path/
        dataset_protocols_path/my_protocol/train/for_background_model.csv
        dataset_protocols_path/my_protocol/train/for_znorm.csv
        dataset_protocols_path/my_protocol/train/for_tnorm.csv
        dataset_protocols_path/my_protocol/dev/for_enrolling.csv
        dataset_protocols_path/my_protocol/dev/for_probing.csv
        dataset_protocols_path/my_protocol/dev/for_matching.csv
        dataset_protocols_path/my_protocol/eval/for_enrolling.csv
        dataset_protocols_path/my_protocol/eval/for_probing.csv
        dataset_protocols_path/my_protocol/eval/for_matching.csv ...

    The ``for_background_model`` file should contain the following columns::

        key,subject_id
        subject1_image1.png,1
        subject1_image2.png,1
        subject2_image1.png,2
        subject2_image2.png,2

    In all the csv files, you can have a column called ``path`` which will be
    used as ``key`` if the ``key`` is not specified. For example::

        path,subject_id
        subject1_image1.png,1
        subject1_image2.png,1
        subject2_image1.png,2
        subject2_image2.png,2

    or::

        path,subject_id,key
        subject1_audio1.wav,1,subject1_audio1_channel1
        subject1_audio1.wav,1,subject1_audio1_channel2
        subject1_audio2.wav,1,subject1_audio2_channel1
        subject1_audio2.wav,1,subject1_audio2_channel2

    The ``key`` column will be used to checkpoint each sample into a unique file and
    must therefore be unique across the whole dataset.

    The ``for_enrolling.csv`` file should contain the following columns::

        key,subject_id,template_id
        subject3_image1.png,3,template_1
        subject3_image2.png,3,template_1
        subject3_image3.png,3,template_2
        subject3_image4.png,3,template_2
        subject4_image1.png,4,template_3
        subject4_image2.png,4,template_3
        subject4_image3.png,4,template_4
        subject4_image4.png,4,template_4

    The ``for_probing.csv`` file should contain the following columns::

        key,subject_id,template_id
        subject5_image1.png,5,template_5
        subject5_image2.png,5,template_5
        subject5_image3.png,5,template_6
        subject5_image4.png,5,template_6
        subject6_image1.png,6,template_7
        subject6_image2.png,6,template_7
        subject6_image3.png,6,template_8
        subject6_image4.png,6,template_8

    Subject identity (``subject_id``) is a unique identifier for one identity (one
    person).
    Template Identity (``template_id``) is an identifier used to group samples when
    they need to be enrolled or scored together.
    :class:`~bob.bio.base.pipelines.BioAlgorithm` will process
    these template.


    By default, each enroll ``template_id`` will be compared against each
    probe ``template_id`` to produce one score per pair. If you want to specify exact
    comparisons (sparse scoring), you can add the ``for_matching.csv`` with the
    following columns::

        enroll_template_id,probe_template_id
        template_1,template_5
        template_2,template_6
        template_3,template_5
        template_3,template_7
        template_4,template_5
        template_4,template_8

    ``for_znorm.csv`` and ``for_tnorm.csv`` files are optional and are used for score
    normalization. See :class:`~bob.bio.base.pipelines.PipelineScoreNorm`.
    ``for_znorm.csv`` has the same format as ``for_probing.csv`` and
    ``for_tnorm.csv`` has the same format as ``for_enrolling.csv``.
    """

    def __init__(
        self,
        *,
        name: str,
        protocol: str,
        dataset_protocols_path: Optional[str] = None,
        transformer: Optional[sklearn.pipeline.Pipeline] = None,
        templates_metadata: Optional[list[str]] = None,
        annotation_type: Optional[str] = None,
        fixed_positions: Optional[dict[str, tuple[float, float]]] = None,
        memory_demanding=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        name
            The name of the database.
        protocol
            Name of the protocol folder in the CSV definition structure.
        dataset_protocol_path
            Path to the CSV files structure (see :ref:`bob.bio.base.database_interface`
            for more info).
        transformer
            An sklearn pipeline or equivalent transformer that handles some light
            preprocessing of the samples (This will always run locally).
        templates_metadata
            Metadata that originate from the samples and must be present in the
            templates (SampleSet) e.g. ``["gender", "age"]``. This should be metadata
            that is common to all the samples in a template.
        annotation_type
            A string describing the annotations passed to the annotation loading
            function
        fixed_positions
            TODO Why is it here? What does it do exactly?
            --> move it when the FaceCrop annotator is implemented correctly.
        memory_demanding
            Flag that indicates that experiments using this should not run on low-mem
            workers.
        """
        if not hasattr(self, "name"):
            self.name = name
        transformer = sklearn.pipeline.make_pipeline(
            sklearn.pipeline.FunctionTransformer(_add_key), transformer
        )
        super().__init__(
            name=name,  # For FileListDatabase
            protocol=protocol,
            dataset_protocols_path=dataset_protocols_path,
            transformer=transformer,
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            memory_demanding=memory_demanding,
            **kwargs,
        )
        if self.list_file("dev", "for_matching") is None:
            self.score_all_vs_all = True
        else:
            self.score_all_vs_all = False

        self.templates_metadata = []
        if templates_metadata is not None:
            self.templates_metadata = templates_metadata

    def list_file(self, group: str, name: str) -> TextIO:
        """Returns a definition file containing one sample per row.

        Overloads ``bob.pipelines`` list_file as the group is a dir.
        """

        try:
            list_file = open_definition_file(
                search_pattern=Path(group) / (name + ".csv"),
                database_name=self.name,
                protocol=self.protocol,
                database_filename=self.dataset_protocols_path.name,
                base_dir=self.dataset_protocols_path.parent,
                subdir=".",
            )
            return list_file
        except FileNotFoundError:
            return None

    def get_reader(self, group: str, name: str) -> Iterable:
        """Returns an :any:`Iterable` containing :class:`Sample` or :class:`SampleSet` objects."""
        key = (self.protocol, group, name)
        if key not in self.readers:
            list_file = self.list_file(group, name)
            self.readers[key] = None
            if list_file is not None:
                self.readers[key] = self.reader_cls(
                    list_file=list_file, transformer=self.transformer
                )

        reader = self.readers[key]
        return reader

    # cached methods should be based on protocol as well
    @functools.lru_cache(maxsize=None)
    def _background_model_samples(self, protocol):
        reader = self.get_reader("train", "for_background_model")
        if reader is None:
            return []
        samples = list(reader)
        validate_bio_samples(samples)
        return samples

    def background_model_samples(self):
        return self._background_model_samples(self.protocol)

    def _sample_sets(self, group, name):
        # we need protocol as input so we can cache the result
        reader = self.get_reader(group, name)
        if reader is None:
            return []
        # create Sample_sets from samples given their unique enroll_template_id/probe_template_id
        samples_grouped_by_template_id = defaultdict(list)
        for sample in reader:
            samples_grouped_by_template_id[sample.template_id].append(sample)
        sample_sets = []
        for (
            template_id,
            samples_for_template_id,
        ) in samples_grouped_by_template_id.items():
            # since all samples inside one sampleset have the same subject_id,
            # we add that as well.
            samples = list(samples_for_template_id)
            subject_id = samples[0].subject_id
            metadata = {
                m: getattr(samples[0], m) for m in self.templates_metadata
            }
            sample_sets.append(
                SampleSet(
                    samples,
                    template_id=template_id,
                    subject_id=subject_id,
                    key=f"template_{template_id}",
                    **metadata,
                )
            )
        validate_bio_samples(sample_sets)
        return sample_sets

    # cached methods should be based on protocol as well
    @functools.lru_cache(maxsize=None)
    def _references(self, protocol, group):  # TODO: protocol
        return self._sample_sets(group, "for_enrolling")

    def references(self, group="dev"):
        return self._references(self.protocol, group)

    def _add_all_references(self, sample_sets, group):
        references = [s.template_id for s in self.references(group)]
        for sample_set in sample_sets:
            sample_set.references = references

    # cached methods should be based on protocol as well
    @functools.lru_cache(maxsize=None)
    def _probes(self, protocol, group):  # TODO: protocol
        sample_sets = self._sample_sets(group, "for_probing")

        # if there are no probes
        if not sample_sets:
            return sample_sets

        # populate .references for each sample set
        matching_file = self.list_file(group, "for_matching")
        if matching_file is None:
            self._add_all_references(sample_sets, group)

        # read the matching file
        else:
            # references is dict where key is probe_template_id and value is a
            # list of enroll_template_ids
            references = defaultdict(list)
            reader = csv.DictReader(matching_file)
            for row in reader:
                references[row["probe_template_id"]].append(
                    row["enroll_template_id"]
                )

            for sample_set in sample_sets:
                sample_set.references = references[sample_set.template_id]

        return sample_sets

    def probes(self, group="dev"):
        return self._probes(self.protocol, group)

    def samples(self, groups=None):
        """Get samples of a certain group

        Parameters
        ----------
        groups : :obj:`str`, optional
            A str or list of str to be used for filtering samples, by default None

        Returns
        -------
        list
            A list containing the samples loaded from csv files.
        """
        groups = check_parameters_for_validity(
            groups, "groups", self.groups(), self.groups()
        )
        all_samples = []

        if "train" in groups:
            all_samples.extend(self.background_model_samples())
            groups.remove("train")

        for grp in groups:
            all_samples.extend(_sample_sets_to_samples(self.references(grp)))
            all_samples.extend(_sample_sets_to_samples(self.probes(grp)))

        # Add znorm samples. Returning znorm samples for one group of dev or
        # eval is enough because they are duplicated.
        for grp in groups:
            all_samples.extend(_sample_sets_to_samples(self.zprobes(grp)))
            break

        # Add tnorm samples.
        all_samples.extend(_sample_sets_to_samples(self.treferences()))

        return all_samples

    def all_samples(self, groups=None):
        return self.samples(groups)

    @functools.lru_cache(maxsize=None)
    def _zprobes(self, protocol, group):
        sample_sets = self._sample_sets("train", "for_znorm")
        if not sample_sets:
            return sample_sets

        self._add_all_references(sample_sets, group)

        return sample_sets

    def zprobes(self, group="dev", proportion=1.0):
        sample_sets = self._zprobes(self.protocol, group)
        if not sample_sets:
            return sample_sets

        sample_sets = sample_sets[: int(len(sample_sets) * proportion)]
        return sample_sets

    @functools.lru_cache(maxsize=None)
    def _treferences(self, protocol):
        sample_sets = self._sample_sets("train", "for_tnorm")
        return sample_sets

    def treferences(self, proportion=1.0):
        sample_sets = self._treferences(self.protocol)
        if not sample_sets:
            return sample_sets

        sample_sets = sample_sets[: int(len(sample_sets) * proportion)]
        return sample_sets


class FileSampleLoader(BaseEstimator, TransformerMixin):
    """Loads file-based samples into :class:`~bob.pipelines.DelayedSample` objects.

    Given the :attr:`sample.path` attribute,``dataset_original_directory`` and an
    ``extension``, this transformer will load lazily the samples from the file.
    The ``data_loader`` is used to load the data.

    The resulting :class:`~bob.pipelines.DelayedSample` objects will call
    ``data_loader`` when their :attr:`data` is accessed.

    This transformer will not access the data files.

    Parameters
    ----------
    data_loader
        A callable to load the sample, given the full path to the file.
    dataset_original_directory
        Path of where the raw data files are stored. This will be prepended to the
        ``path`` attribute of the samples.
    extension
        File extension of the raw data files. This will be appended to the ``path``
        attribute of the samples.
    """

    def __init__(
        self,
        data_loader: Callable[[str], Any],
        dataset_original_directory: str = "",
        extension: str = "",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_loader = data_loader
        self.dataset_original_directory = dataset_original_directory
        self.extension = extension

    def transform(self, samples: list[Sample]) -> list[DelayedSample]:
        """Prepares the data into :class:`~bob.pipelines.DelayedSample` objects.

        Transforms :class:`~bob.pipelines.Sample` objects with a ``path`` attribute to
        :class:`~bob.pipelines.DelayedSample` with data ready to be loaded (lazily) by
        :attr:`data_loader`.

        When needed (access to the :class:`DelayedSample`\\ 's :attr:`data` attribute),
        :attr:`data_loader` will be called with the path (extended with
        :attr:`original_directory` and :attr:`extension`) as argument.

        Parameters
        ----------
        samples
            :class:`~bob.pipelines.Sample` objects with their ``path`` attribute
            containing a path to a file to load.
        """
        output = []
        for sample in samples:
            path = getattr(sample, "path")
            delayed_sample = DelayedSample(
                functools.partial(
                    self.data_loader,
                    os.path.join(
                        # we append ./ to path to make sure that the path is
                        # relative to the dataset_original_directory
                        self.dataset_original_directory,
                        f"./{path + self.extension}",
                    ),
                ),
                parent=sample,
            )
            output.append(delayed_sample)
        return output

    def _more_tags(self):
        return {"requires_fit": False}


class AnnotationsLoader(TransformerMixin, BaseEstimator):
    """Prepares annotations to be loaded from a path in ``delayed_attributes``.

    Metadata loader that loads samples' annotations using
    :py:func:`~bob.bio.base.utils.annotations.read_annotation_file`. This assumes that
    the annotation files follows the same folder structure and naming as the raw data
    files. Although, the base location and the extension can vary from those and is
    specified by :attr:`annotation_directory` and :attr:`annotation_extension`.

    Parameters
    ----------
    annotation_directory
        Path where the annotations are stored.

    annotation_extension : str
        Extension of the annotations.

    annotation_type : str
        Annotations type passed to
        :func:`~bob.bio.base.utils.annotations.read_annotation_file`.

    """

    def __init__(
        self,
        annotation_directory: Optional[str] = None,
        annotation_extension: str = ".json",
        annotation_type: str = "json",
    ):
        self.annotation_directory = annotation_directory
        self.annotation_extension = annotation_extension
        self.annotation_type = annotation_type

    def transform(self, X: list[DelayedSample]) -> list[DelayedSample]:
        """Edits the samples to lazily load annotations files.

        Parameters
        ----------
        X
            The samples to augment.
        """
        if self.annotation_directory is None:
            return None

        annotated_samples = []
        for x in X:
            # we use .key here because .path might not be unique for all
            # samples. Also, the ``bob bio annotate-samples`` command dictates
            # how annotations are stored.
            annotation_file = os.path.join(
                self.annotation_directory, x.key + self.annotation_extension
            )

            annotated_samples.append(
                DelayedSample.from_sample(
                    x,
                    delayed_attributes=dict(
                        annotations=lambda: read_annotation_file(
                            annotation_file, self.annotation_type
                        )
                    ),
                )
            )

        return annotated_samples

    def _more_tags(self):
        return {"requires_fit": False}
