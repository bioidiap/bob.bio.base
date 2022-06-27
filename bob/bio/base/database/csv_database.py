import csv
import functools
import itertools
import os

from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin

from bob.bio.base.pipelines.abstract_classes import Database
from bob.extension.download import list_dir, search_file
from bob.pipelines import (
    DelayedSample,
    FileListDatabase,
    SampleSet,
    check_parameters_for_validity,
)

from ..utils.annotations import read_annotation_file


def _sample_sets_to_samples(sample_sets):
    return functools.reduce(
        lambda x, y: x + y, (s.samples for s in sample_sets), []
    )


def validate_bio_samples(samples):
    """Validates Samples or SampleSets for backwards compatibility reasons."""
    for sample in samples:

        if isinstance(sample, SampleSet):
            validate_bio_samples(sample.samples)

            if not hasattr(sample, "template_id"):
                raise ValueError(
                    f"SampleSet must have a template_id attribute, got {sample}"
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

    The ``key`` column will be used to checkpoint the each sample into a unique
    file.

    The ``for_enrolling.csv`` file should contain the following columns::

        key,subject_id,enroll_template_id
        subject3_image1.png,3,template_1
        subject3_image2.png,3,template_1
        subject3_image3.png,3,template_2
        subject3_image4.png,3,template_2
        subject4_image1.png,4,template_3
        subject4_image2.png,4,template_3
        subject4_image3.png,4,template_4
        subject4_image4.png,4,template_4

    The ``for_probing.csv`` file should contain the following columns::

        key,subject_id,probe_template_id
        subject5_image1.png,5,template_5
        subject5_image2.png,5,template_5
        subject5_image3.png,5,template_6
        subject5_image4.png,5,template_6
        subject6_image1.png,6,template_7
        subject6_image2.png,6,template_7
        subject6_image3.png,6,template_8
        subject6_image4.png,6,template_8

    By default, each enroll_template_id will be compared against each
    probe_template_id to produce one score. If you want to specify exact
    comparisons (sparse scoring), you can add the ``for_matching.csv`` with the
    following columns::

        enroll_template_id,probe_template_id
        template_1,template_5
        template_2,template_6
        template_3,template_7
        template_4,template_8

    ``for_znorm.csv`` and ``for_tnorm.csv`` files are optional.
    ``for_znorm.csv`` has the same format as ``for_probing.csv`` and
    ``for_tnorm.csv`` has the same format as ``for_enrolling.csv``.
    """

    def __init__(
        self,
        *,
        name,
        protocol,
        dataset_protocols_path,
        transformer=None,
        annotation_type=None,
        fixed_positions=None,
        memory_demanding=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            protocol=protocol,
            dataset_protocols_path=dataset_protocols_path,
            transformer=transformer,
            annotation_type=annotation_type,
            fixed_positions=fixed_positions,
            memory_demanding=memory_demanding,
            **kwargs,
        )
        if self.list_file("dev", "for_matching.csv") is None:
            self.score_all_vs_all = True
        else:
            self.score_all_vs_all = False

    def list_file(self, group, name):
        list_file = search_file(
            self.dataset_protocols_path,
            os.path.join(self.protocol, group, name + ".csv"),
        )
        return list_file

    def get_reader(self, group, name):
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

    def groups(self):
        names = list_dir(
            self.dataset_protocols_path,
            self.protocol,
            folders=True,
            files=False,
        )

        return names

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

    def _sample_sets(self, group, name, attr):
        # we need protocol as input so we can cache the result
        reader = self.get_reader(group, name)
        if reader is None:
            return []
        samples = list(reader)
        # create Sample_sets from samples given their unique enroll_template_id/probe_template_id
        samples_grouped_by_template_id = itertools.groupby(
            samples, lambda x: getattr(x, attr)
        )
        sample_sets = []
        for (
            template_id,
            samples_for_template_id,
        ) in samples_grouped_by_template_id:
            sample_sets.append(
                SampleSet(
                    list(samples_for_template_id), template_id=template_id
                )
            )
        validate_bio_samples(sample_sets)
        return sample_sets

    # cached methods should be based on protocol as well
    @functools.lru_cache(maxsize=None)
    def _references(self, protocol, group):
        return self._sample_sets(
            group, "for_enrolling", attr="enroll_template_id"
        )

    def references(self, group="dev"):
        return self._references(self.protocol, group)

    def _add_all_references(self, sample_sets, group):
        references = [s.template_id for s in self.references(group)]
        for sample_set in sample_sets:
            sample_set.references = references

    # cached methods should be based on protocol as well
    @functools.lru_cache(maxsize=None)
    def _probes(self, protocol, group):
        sample_sets = self._sample_sets(
            group, "for_probing", attr="probe_template_id"
        )

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
            with open(matching_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    references[row["probe_template_id"]].append(
                        row["enroll_template_id"]
                    )

            for sample_set in sample_sets:
                sample_set.references = references[sample_set.template_id]

        return sample_sets

    def probes(self, group):
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
        sample_sets = self._sample_sets(
            "train", "for_znorm", attr="probe_template_id"
        )
        if not sample_sets:
            return sample_sets

        self._add_all_references(sample_sets, group)

        return sample_sets

    def zprobes(self, group, proportion=1.0):
        sample_sets = self._zprobes(self.protocol, group)
        if not sample_sets:
            return sample_sets

        sample_sets = sample_sets[: int(len(sample_sets) * proportion)]
        return sample_sets

    @functools.lru_cache(maxsize=None)
    def _treferences(self, protocol):
        sample_sets = self._sample_sets(
            "train", "for_tnorm", attr="enroll_template_id"
        )
        return sample_sets

    def treferences(self, proportion=1.0):
        sample_sets = self._treferences(self.protocol)
        if not sample_sets:
            return sample_sets

        sample_sets = sample_sets[: int(len(sample_sets) * proportion)]
        return sample_sets


class FileSampleLoader(BaseEstimator, TransformerMixin):
    """
    Base class that converts the lines of a CSV file, like the one below to
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`

    .. code-block:: text

       PATH,REFERENCE_ID
       path_1,reference_id_1
       path_2,reference_id_2
       path_i,reference_id_j
       ...

    Parameters
    ----------

        data_loader:
            A callable to load the sample, given the ``sample.path``.

        dataset_original_directory: str
            Path of where data is stored

        extension: str
            Default file extension

    """

    def __init__(
        self,
        data_loader,
        dataset_original_directory="",
        extension="",
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.data_loader = data_loader
        self.dataset_original_directory = dataset_original_directory
        self.extension = extension

    def transform(self, samples):
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

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {"requires_fit": False}


class AnnotationsLoader(TransformerMixin, BaseEstimator):
    """
    Metadata loader that loads annotations in the Idiap format using the function
    :any:`read_annotation_file`

    Parameters
    ----------

    annotation_directory: str
        Path where the annotations are store

    annotation_extension: str
        Extension of the annotations

    annotation_type: str
        Annotations type

    """

    def __init__(
        self,
        annotation_directory=None,
        annotation_extension=".json",
        annotation_type="json",
    ):
        self.annotation_directory = annotation_directory
        self.annotation_extension = annotation_extension
        self.annotation_type = annotation_type

    def transform(self, X):
        if self.annotation_directory is None:
            return None

        annotated_samples = []
        for x in X:

            # since the file id is equal to the file name, we can simply use it
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

    def fit(self, X, y=None):
        return self

    def _more_tags(self):
        return {"requires_fit": False}
