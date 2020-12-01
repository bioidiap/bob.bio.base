#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import os
from bob.pipelines import Sample, DelayedSample, SampleSet
from bob.db.base.utils import check_parameters_for_validity
import csv
import bob.io.base
import functools
from abc import ABCMeta, abstractmethod
import numpy as np
import itertools
import logging
import bob.db.base

from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import Database

logger = logging.getLogger(__name__)


#####
# ANNOTATIONS LOADERS
####
class AnnotationsLoader:
    """
    Load annotations in the Idiap format
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

    def __call__(self, row, header=None):
        if self.annotation_directory is None:
            return None

        path = row[0]

        # since the file id is equal to the file name, we can simply use it
        annotation_file = os.path.join(
            self.annotation_directory, path + self.annotation_extension
        )

        # return the annotations as read from file
        annotation = {
            "annotations": bob.db.base.read_annotation_file(
                annotation_file, self.annotation_type
            )
        }
        return annotation


#######
# SAMPLE LOADERS
# CONVERT CSV LINES TO SAMPLES
#######


class CSVBaseSampleLoader(metaclass=ABCMeta):
    """
    Convert CSV files in the format below to either a list of
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`

    .. code-block:: text

       PATH,REFERENCE_ID
       path_1,reference_id_1
       path_2,reference_id_2
       path_i,reference_id_j
       ...

    .. note::
       This class should be extended

    Parameters
    ----------

        data_loader:
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

        extension:
            The file extension

    """

    def __init__(
        self,
        data_loader,
        metadata_loader=None,
        dataset_original_directory="",
        extension="",
    ):
        self.data_loader = data_loader
        self.extension = extension
        self.dataset_original_directory = dataset_original_directory
        self.metadata_loader = metadata_loader

    @abstractmethod
    def __call__(self, filename):
        pass

    @abstractmethod
    def convert_row_to_sample(self, row, header):
        pass

    def convert_samples_to_samplesets(
        self, samples, group_by_reference_id=True, references=None
    ):
        if group_by_reference_id:

            # Grouping sample sets
            sample_sets = dict()
            for s in samples:
                if s.reference_id not in sample_sets:
                    sample_sets[s.reference_id] = (
                        SampleSet([s], parent=s)
                        if references is None
                        else SampleSet([s], parent=s, references=references)
                    )
                else:
                    sample_sets[s.reference_id].append(s)
            return list(sample_sets.values())

        else:
            return (
                [SampleSet([s], parent=s) for s in samples]
                if references is None
                else [SampleSet([s], parent=s, references=references) for s in samples]
            )


class CSVToSampleLoader(CSVBaseSampleLoader):
    """
    Simple mechanism to convert CSV files in the format below to either a list of
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`
    """

    def check_header(self, header):
        """
        A header should have at least "reference_id" AND "PATH"
        """
        header = [h.lower() for h in header]
        if not "reference_id" in header:
            raise ValueError(
                "The field `reference_id` is not available in your dataset."
            )

        if not "path" in header:
            raise ValueError("The field `path` is not available in your dataset.")

    def __call__(self, filename):

        with open(filename) as cf:
            reader = csv.reader(cf)
            header = next(reader)

            self.check_header(header)
            return [self.convert_row_to_sample(row, header) for row in reader]

    def convert_row_to_sample(self, row, header):
        path = row[0]
        reference_id = row[1]

        kwargs = dict([[str(h).lower(), r] for h, r in zip(header[2:], row[2:])])

        if self.metadata_loader is not None:
            metadata = self.metadata_loader(row, header=header)
            kwargs.update(metadata)

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            reference_id=reference_id,
            **kwargs,
        )


class LSTToSampleLoader(CSVBaseSampleLoader):
    """
    Simple mechanism to convert LST files in the format below to either a list of
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`
    """

    def __call__(self, filename):

        with open(filename) as cf:
            reader = csv.reader(cf, delimiter=" ")
            samples = []
            for row in reader:
                if row[0][0] == "#":
                    continue
                samples.append(self.convert_row_to_sample(row))

            return samples

    def convert_row_to_sample(self, row, header=None):

        if len(row) == 4:
            path = row[0]
            compare_reference_id = row[1]
            reference_id = str(row[3])
            kwargs = {"compare_reference_id": str(compare_reference_id)}
        else:
            path = row[0]
            reference_id = str(row[1])
            kwargs = dict()
            if len(row) == 3:
                subject = row[2]
                kwargs = {"subject": str(subject)}

        if self.metadata_loader is not None:
            metadata = self.metadata_loader(row, header=header)
            kwargs.update(metadata)

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            reference_id=reference_id,
            **kwargs,
        )


#####
# DATABASE INTERFACES
#####


class CSVDatasetDevEval(Database):
    """
    Generic filelist dataset for :any:` bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline` pipeline.
    Check :any:`vanilla_biometrics_features` for more details about the Vanilla Biometrics Dataset
    interface.

    To create a new dataset, you need to provide a directory structure similar to the one below:

    .. code-block:: text

       my_dataset/
       my_dataset/my_protocol/norm/train_world.csv
       my_dataset/my_protocol/dev/for_models.csv
       my_dataset/my_protocol/dev/for_probes.csv
       my_dataset/my_protocol/eval/for_models.csv
       my_dataset/my_protocol/eval/for_probes.csv
       ...


    In the above directory structure, inside of `my_dataset` should contain the directories with all
    evaluation protocols this dataset might have.
    Inside of the `my_protocol` directory should contain at least two csv files:

     - for_models.csv
     - for_probes.csv


    Those csv files should contain in each row i-) the path to raw data and ii-) the reference_id label
    for enrollment (:any:`bob.bio.base.pipelines.vanilla_biometrics.Database.references`) and
    probing (:any:`bob.bio.base.pipelines.vanilla_biometrics.Database.probes`).
    The structure of each CSV file should be as below:

    .. code-block:: text

       PATH,reference_id
       path_1,reference_id_1
       path_2,reference_id_2
       path_i,reference_id_j
       ...


    You might want to ship metadata within your Samples (e.g gender, age, annotation, ...)
    To do so is simple, just do as below:

    .. code-block:: text

       PATH,reference_id,METADATA_1,METADATA_2,METADATA_k
       path_1,reference_id_1,A,B,C
       path_2,reference_id_2,A,B,1
       path_i,reference_id_j,2,3,4
       ...


    The files `my_dataset/my_protocol/train.csv/eval_enroll.csv` and `my_dataset/my_protocol/train.csv/eval_probe.csv`
    are optional and it is used in case a protocol contains data for evaluation.

    Finally, the content of the file `my_dataset/my_protocol/train.csv` is used in the case a protocol
    contains data for training (`bob.bio.base.pipelines.vanilla_biometrics.Database.background_model_samples`)

    Parameters
    ----------

        dataset_path: str
          Absolute path of the dataset protocol description

        protocol_na,e: str
          The name of the protocol

        csv_to_sample_loader: :any:`bob.bio.base.database.CSVBaseSampleLoader`
            Base class that whose objective is to generate :any:`bob.pipelines.Sample`
            and/or :any:`bob.pipelines.SampleSet` from csv rows

    """

    def __init__(
        self,
        dataset_protocol_path,
        protocol_name,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load,
            metadata_loader=None,
            dataset_original_directory="",
            extension="",
        ),
        is_sparse=False,
    ):
        self.dataset_protocol_path = dataset_protocol_path
        self.is_sparse = is_sparse

        def get_paths():

            if not os.path.exists(dataset_protocol_path):
                raise ValueError(f"The path `{dataset_protocol_path}` was not found")

            # TODO: Unzip file if dataset path is a zip
            protocol_path = os.path.join(dataset_protocol_path, protocol_name)
            if not os.path.exists(protocol_path):
                raise ValueError(f"The protocol `{protocol_name}` was not found")

            def path_discovery(option1, option2):
                return option1 if os.path.exists(option1) else option2

            # Here we are handling the legacy
            train_csv = path_discovery(
                os.path.join(protocol_path, "norm", "train_world.lst"),
                os.path.join(protocol_path, "norm", "train_world.csv"),
            )

            dev_enroll_csv = path_discovery(
                os.path.join(protocol_path, "dev", "for_models.lst"),
                os.path.join(protocol_path, "dev", "for_models.csv"),
            )

            legacy_probe = "for_scores.lst" if self.is_sparse else "for_probes.lst"
            dev_probe_csv = path_discovery(
                os.path.join(protocol_path, "dev", legacy_probe),
                os.path.join(protocol_path, "dev", "for_probes.csv"),
            )

            eval_enroll_csv = path_discovery(
                os.path.join(protocol_path, "eval", "for_models.lst"),
                os.path.join(protocol_path, "eval", "for_models.csv"),
            )

            eval_probe_csv = path_discovery(
                os.path.join(protocol_path, "eval", legacy_probe),
                os.path.join(protocol_path, "eval", "for_probes.csv"),
            )

            # The minimum required is to have `dev_enroll_csv` and `dev_probe_csv`
            train_csv = train_csv if os.path.exists(train_csv) else None

            # Eval
            eval_enroll_csv = (
                eval_enroll_csv if os.path.exists(eval_enroll_csv) else None
            )
            eval_probe_csv = eval_probe_csv if os.path.exists(eval_probe_csv) else None

            # Dev
            if not os.path.exists(dev_enroll_csv):
                raise ValueError(
                    f"The file `{dev_enroll_csv}` is required and it was not found"
                )

            if not os.path.exists(dev_probe_csv):
                raise ValueError(
                    f"The file `{dev_probe_csv}` is required and it was not found"
                )
            dev_enroll_csv = dev_enroll_csv
            dev_probe_csv = dev_probe_csv

            return (
                train_csv,
                dev_enroll_csv,
                dev_probe_csv,
                eval_enroll_csv,
                eval_probe_csv,
            )

        (
            self.train_csv,
            self.dev_enroll_csv,
            self.dev_probe_csv,
            self.eval_enroll_csv,
            self.eval_probe_csv,
        ) = get_paths()

        def get_dict_cache():
            cache = dict()
            cache["train"] = None
            cache["dev_enroll_csv"] = None
            cache["dev_probe_csv"] = None
            cache["eval_enroll_csv"] = None
            cache["eval_probe_csv"] = None
            return cache

        self.cache = get_dict_cache()
        self.csv_to_sample_loader = csv_to_sample_loader

    def background_model_samples(self):
        self.cache["train"] = (
            self.csv_to_sample_loader(self.train_csv)
            if self.cache["train"] is None
            else self.cache["train"]
        )

        return self.cache["train"]

    def _get_samplesets(
        self, group="dev", purpose="enroll", group_by_reference_id=False
    ):

        fetching_probes = False
        if purpose == "enroll":
            cache_label = "dev_enroll_csv" if group == "dev" else "eval_enroll_csv"
        else:
            fetching_probes = True
            cache_label = "dev_probe_csv" if group == "dev" else "eval_probe_csv"

        if self.cache[cache_label] is not None:
            return self.cache[cache_label]

        # Getting samples from CSV
        samples = self.csv_to_sample_loader(self.__dict__[cache_label])

        references = None
        if fetching_probes and self.is_sparse:

            # Checking if `is_sparse` was set properly
            if len(samples) > 0 and not hasattr(samples[0], "compare_reference_id"):
                ValueError(
                    f"Attribute `compare_reference_id` not found in `{samples[0]}`."
                    "Make sure this attribute exists in your dataset if `is_sparse=True`"
                )

            sparse_samples = dict()
            for s in samples:
                if s.key in sparse_samples:
                    sparse_samples[s.key].references.append(s.compare_reference_id)
                else:
                    s.references = [s.compare_reference_id]
                    sparse_samples[s.key] = s
            samples = sparse_samples.values()
        else:
            if fetching_probes:
                references = list(
                    set([s.reference_id for s in self.references(group=group)])
                )

        sample_sets = self.csv_to_sample_loader.convert_samples_to_samplesets(
            samples, group_by_reference_id=group_by_reference_id, references=references,
        )

        self.cache[cache_label] = sample_sets

        return self.cache[cache_label]

    def references(self, group="dev"):
        return self._get_samplesets(
            group=group, purpose="enroll", group_by_reference_id=True
        )

    def probes(self, group="dev"):
        return self._get_samplesets(
            group=group, purpose="probe", group_by_reference_id=False
        )

    def all_samples(self, groups=None):
        """
        Reads and returns all the samples in `groups`.

        Parameters
        ----------
        groups: list or None
            Groups to consider ('train', 'dev', and/or 'eval'). If `None` is
            given, returns the samples from all groups.

        Returns
        -------
        samples: list
            List of :class:`bob.pipelines.Sample` objects.
        """
        valid_groups = ["train"]
        if self.dev_enroll_csv and self.dev_probe_csv:
            valid_groups.append("dev")
        if self.eval_enroll_csv and self.eval_probe_csv:
            valid_groups.append("eval")
        groups = check_parameters_for_validity(
            parameters=groups,
            parameter_description="groups",
            valid_parameters=valid_groups,
            default_parameters=valid_groups,
        )

        samples = []

        # Get train samples (background_model_samples returns a list of samples)
        if "train" in groups:
            samples = samples + self.background_model_samples()
            groups.remove("train")

        # Get enroll and probe samples
        for group in groups:
            for purpose in ("enroll", "probe"):
                label = f"{group}_{purpose}_csv"
                samples = samples + self.csv_to_sample_loader(self.__dict__[label])
        return samples

    def groups(self):
        """This function returns the list of groups for this database.

        Returns
        -------

        [str]
          A list of groups
        """

        # We always have dev-set
        groups = ["dev"]

        if self.train_csv is not None:
            groups.append("train")

        if self.eval_enroll_csv is not None:
            groups.append("eval")

        return groups


class CSVDatasetCrossValidation:
    """
    Generic filelist dataset for :any:`bob.bio.base.pipelines.vanilla_biometrics.VanillaBiometricsPipeline` pipeline that
    handles **CROSS VALIDATION**.

    Check :any:`vanilla_biometrics_features` for more details about the Vanilla Biometrics Dataset
    interface.


    This interface will take one `csv_file` as input and split into i-) data for training and
    ii-) data for testing.
    The data for testing will be further split in data for enrollment and data for probing.
    The input CSV file should be casted in the following format:

    .. code-block:: text

       PATH,reference_id
       path_1,reference_id_1
       path_2,reference_id_2
       path_i,reference_id_j
       ...

    Parameters
    ----------

    csv_file_name: str
      CSV file containing all the samples from your database

    random_state: int
      Pseudo-random number generator seed

    test_size: float
      Percentage of the reference_ids used for testing

    samples_for_enrollment: float
      Number of samples used for enrollment

    csv_to_sample_loader: :any:`bob.bio.base.database.CSVBaseSampleLoader`
        Base class that whose objective is to generate :any:`bob.pipelines.Sample`
        and/or :any:`bob.pipelines.SampleSet` from csv rows

    """

    def __init__(
        self,
        csv_file_name="metadata.csv",
        random_state=0,
        test_size=0.8,
        samples_for_enrollment=1,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load, dataset_original_directory="", extension=""
        ),
    ):
        def get_dict_cache():
            cache = dict()
            cache["train"] = None
            cache["dev_enroll_csv"] = None
            cache["dev_probe_csv"] = None
            return cache

        self.random_state = random_state
        self.cache = get_dict_cache()
        self.csv_to_sample_loader = csv_to_sample_loader
        self.csv_file_name = csv_file_name
        self.samples_for_enrollment = samples_for_enrollment
        self.test_size = test_size

        if self.test_size < 0 and self.test_size > 1:
            raise ValueError(
                f"`test_size` should be between 0 and 1. {test_size} is provided"
            )

    def _do_cross_validation(self):

        # Shuffling samples by reference_id
        samples_by_reference_id = group_samples_by_reference_id(
            self.csv_to_sample_loader(self.csv_file_name)
        )
        reference_ids = list(samples_by_reference_id.keys())
        np.random.seed(self.random_state)
        np.random.shuffle(reference_ids)

        # Getting the training data
        n_samples_for_training = len(reference_ids) - int(
            self.test_size * len(reference_ids)
        )
        self.cache["train"] = list(
            itertools.chain(
                *[
                    samples_by_reference_id[s]
                    for s in reference_ids[0:n_samples_for_training]
                ]
            )
        )

        # Splitting enroll and probe
        self.cache["dev_enroll_csv"] = []
        self.cache["dev_probe_csv"] = []
        for s in reference_ids[n_samples_for_training:]:
            samples = samples_by_reference_id[s]
            if len(samples) < self.samples_for_enrollment:
                raise ValueError(
                    f"Not enough samples ({len(samples)}) for enrollment for the reference_id {s}"
                )

            # Enrollment samples
            self.cache["dev_enroll_csv"].append(
                self.csv_to_sample_loader.convert_samples_to_samplesets(
                    samples[0 : self.samples_for_enrollment]
                )[0]
            )

            self.cache[
                "dev_probe_csv"
            ] += self.csv_to_sample_loader.convert_samples_to_samplesets(
                samples[self.samples_for_enrollment :],
                group_by_reference_id=False,
                references=reference_ids[n_samples_for_training:],
            )

    def _load_from_cache(self, cache_key):
        if self.cache[cache_key] is None:
            self._do_cross_validation()
        return self.cache[cache_key]

    def background_model_samples(self):
        return self._load_from_cache("train")

    def references(self, group="dev"):
        return self._load_from_cache("dev_enroll_csv")

    def probes(self, group="dev"):
        return self._load_from_cache("dev_probe_csv")

    def all_samples(self, groups=None):
        """
        Reads and returns all the samples in `groups`.

        Parameters
        ----------
        groups: list or None
            Groups to consider ('train' and/or 'dev'). If `None` is given,
            returns the samples from all groups.

        Returns
        -------
        samples: list
            List of :class:`bob.pipelines.Sample` objects.
        """
        valid_groups = ["train", "dev"]
        groups = check_parameters_for_validity(
            parameters=groups,
            parameter_description="groups",
            valid_parameters=valid_groups,
            default_parameters=valid_groups,
        )

        samples = []

        # Get train samples (background_model_samples returns a list of samples)
        if "train" in groups:
            samples = samples + self.background_model_samples()
            groups.remove("train")

        # Get enroll and probe samples
        for group in groups:
            samples = samples + [s for s_set in self.references(group) for s in s_set]
            samples = samples + [s for s_set in self.probes(group) for s in s_set]
        return samples


def group_samples_by_reference_id(samples):

    # Grouping sample sets
    samples_by_reference_id = dict()
    for s in samples:
        if s.reference_id not in samples_by_reference_id:
            samples_by_reference_id[s.reference_id] = []
        samples_by_reference_id[s.reference_id].append(s)
    return samples_by_reference_id
