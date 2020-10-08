#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import os
from bob.pipelines import Sample, DelayedSample, SampleSet
import csv
import bob.io.base
import functools
from abc import ABCMeta, abstractmethod
import numpy as np
import itertools


class CSVBaseSampleLoader(metaclass=ABCMeta):
    """
    Convert CSV files in the format below to either a list of
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`

    .. code-block:: text

       PATH,SUBJECT
       path_1,subject_1
       path_2,subject_2
       path_i,subject_j
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

    def __init__(self, data_loader, dataset_original_directory="", extension=""):
        self.data_loader = data_loader
        self.extension = extension
        self.dataset_original_directory = dataset_original_directory
        self.excluding_attributes = ["_data", "load", "key"]

    @abstractmethod
    def __call__(self, filename):
        pass

    @abstractmethod
    def convert_row_to_sample(self, row, header):
        pass

    @abstractmethod
    def convert_samples_to_samplesets(
        self, samples, group_by_subject=True, references=None
    ):
        pass


class CSVToSampleLoader(CSVBaseSampleLoader):
    """
    Simple mechanism to convert CSV files in the format below to either a list of
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`
    """

    def __call__(self, filename):
        def check_header(header):
            """
            A header should have at least "SUBJECT" AND "PATH"
            """
            header = [h.lower() for h in header]
            if not "subject" in header:
                raise ValueError(
                    "The field `subject` is not available in your dataset."
                )

            if not "path" in header:
                raise ValueError("The field `path` is not available in your dataset.")

        with open(filename) as cf:
            reader = csv.reader(cf)
            header = next(reader)

            check_header(header)
            return [self.convert_row_to_sample(row, header) for row in reader]

    def convert_row_to_sample(self, row, header):
        path = row[0]
        subject = row[1]
        kwargs = dict([[h, r] for h, r in zip(header[2:], row[2:])])
        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(self.dataset_original_directory, path + self.extension),
            ),
            key=path,
            subject=subject,
            **kwargs,
        )

    def convert_samples_to_samplesets(
        self, samples, group_by_subject=True, references=None
    ):
        def get_attribute_from_sample(sample):
            return dict(
                [
                    [attribute, sample.__dict__[attribute]]
                    for attribute in list(sample.__dict__.keys())
                    if attribute not in self.excluding_attributes
                ]
            )

        if group_by_subject:

            # Grouping sample sets
            sample_sets = dict()
            for s in samples:
                if s.subject not in sample_sets:
                    sample_sets[s.subject] = SampleSet(
                        [s], **get_attribute_from_sample(s)
                    )
                else:
                    sample_sets[s.subject].append(s)
            return list(sample_sets.values())

        else:
            return [
                SampleSet([s], **get_attribute_from_sample(s), references=references)
                for s in samples
            ]


class CSVDatasetDevEval:
    """
    Generic filelist dataset for :any:`bob.bio.base.pipelines.VanillaBiometrics` pipeline.
    Check :ref:`vanilla_biometrics_features` for more details about the Vanilla Biometrics Dataset
    interface.

    To create a new dataset, you need to provide a directory structure similar to the one below:

    .. code-block:: text

       my_dataset/
       my_dataset/my_protocol/
       my_dataset/my_protocol/train.csv
       my_dataset/my_protocol/train.csv/dev_enroll.csv
       my_dataset/my_protocol/train.csv/dev_probe.csv
       my_dataset/my_protocol/train.csv/eval_enroll.csv
       my_dataset/my_protocol/train.csv/eval_probe.csv
       ...


    In the above directory structure, inside of `my_dataset` should contain the directories with all
    evaluation protocols this dataset might have.
    Inside of the `my_protocol` directory should contain at least two csv files:

     - dev_enroll.csv
     - dev_probe.csv


    Those csv files should contain in each row i-) the path to raw data and ii-) the subject label
    for enrollment (:ref:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.Database.references`) and
    probing (:ref:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.Database.probes`).
    The structure of each CSV file should be as below:

    .. code-block:: text

       PATH,SUBJECT
       path_1,subject_1
       path_2,subject_2
       path_i,subject_j
       ...

    
    You might want to ship metadata within your Samples (e.g gender, age, annotation, ...)
    To do so is simple, just do as below:

    .. code-block:: text

       PATH,SUBJECT,METADATA_1,METADATA_2,METADATA_k
       path_1,subject_1,A,B,C
       path_2,subject_2,A,B,1
       path_i,subject_j,2,3,4
       ...


    The files `my_dataset/my_protocol/train.csv/eval_enroll.csv` and `my_dataset/my_protocol/train.csv/eval_probe.csv`
    are optional and it is used in case a protocol contains data for evaluation.
    
    Finally, the content of the file `my_dataset/my_protocol/train.csv` is used in the case a protocol
    contains data for training (:ref:`bob.bio.base.pipelines.vanilla_biometrics.abstract_classes.Database.background_model_samples`)

    Parameters
    ----------

        dataset_path: str
          Absolute path of the dataset protocol description

        protocol_na,e: str
          The name of the protocol

        csv_to_sample_loader: :any:`CSVBaseSampleLoader`
            Base class that whose objective is to generate :any:`bob.pipelines.Samples`
            and/or :any:`bob.pipelines.SampleSet` from csv rows

    """

    def __init__(
        self,
        dataset_protocol_path,
        protocol_name,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load, dataset_original_directory="", extension=""
        ),
    ):
        def get_paths():

            if not os.path.exists(dataset_protocol_path):
                raise ValueError(f"The path `{dataset_protocol_path}` was not found")

            # TODO: Unzip file if dataset path is a zip
            protocol_path = os.path.join(dataset_protocol_path, protocol_name)
            if not os.path.exists(protocol_path):
                raise ValueError(f"The protocol `{protocol_name}` was not found")

            train_csv = os.path.join(protocol_path, "train.csv")
            dev_enroll_csv = os.path.join(protocol_path, "dev_enroll.csv")
            dev_probe_csv = os.path.join(protocol_path, "dev_probe.csv")
            eval_enroll_csv = os.path.join(protocol_path, "eval_enroll.csv")
            eval_probe_csv = os.path.join(protocol_path, "eval_probe.csv")

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

    def _get_samplesets(self, group="dev", purpose="enroll", group_by_subject=False):

        fetching_probes = False
        if purpose == "enroll":
            cache_label = "dev_enroll_csv" if group == "dev" else "eval_enroll_csv"
        else:
            fetching_probes = True
            cache_label = "dev_probe_csv" if group == "dev" else "eval_probe_csv"

        if self.cache[cache_label] is not None:
            return self.cache[cache_label]

        references = None
        if fetching_probes:
            references = list(set([s.subject for s in self.references(group=group)]))

        samples = self.csv_to_sample_loader(self.__dict__[cache_label])

        sample_sets = self.csv_to_sample_loader.convert_samples_to_samplesets(
            samples, group_by_subject=group_by_subject, references=references
        )

        self.cache[cache_label] = sample_sets

        return self.cache[cache_label]

    def references(self, group="dev"):
        return self._get_samplesets(
            group=group, purpose="enroll", group_by_subject=True
        )

    def probes(self, group="dev"):
        return self._get_samplesets(
            group=group, purpose="probe", group_by_subject=False
        )


class CSVDatasetCrossValidation:
    """
    Generic filelist dataset for :any:`bob.bio.base.pipelines.VanillaBiometrics` pipeline that 
    handles **CROSS VALIDATION**.

    Check :ref:`vanilla_biometrics_features` for more details about the Vanilla Biometrics Dataset
    interface.


    This interface will take one `csv_file` as input and split into i-) data for training and
    ii-) data for testing.
    The data for testing will be further split in data for enrollment and data for probing.
    The input CSV file should be casted in the following format:

    .. code-block:: text

       PATH,SUBJECT
       path_1,subject_1
       path_2,subject_2
       path_i,subject_j
       ...

    Parameters
    ----------

    csv_file_name: str
      CSV file containing all the samples from your database

    random_state: int
      Pseudo-random number generator seed

    test_size: float
      Percentage of the subjects used for testing

    samples_for_enrollment: float
      Number of samples used for enrollment

    csv_to_sample_loader: :any:`CSVBaseSampleLoader`
        Base class that whose objective is to generate :any:`bob.pipelines.Samples`
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

        # Shuffling samples by subject
        samples_by_subject = group_samples_by_subject(
            self.csv_to_sample_loader(self.csv_file_name)
        )
        subjects = list(samples_by_subject.keys())
        np.random.seed(self.random_state)
        np.random.shuffle(subjects)

        # Getting the training data
        n_samples_for_training = len(subjects) - int(self.test_size * len(subjects))
        self.cache["train"] = list(
            itertools.chain(
                *[samples_by_subject[s] for s in subjects[0:n_samples_for_training]]
            )
        )

        # Splitting enroll and probe
        self.cache["dev_enroll_csv"] = []
        self.cache["dev_probe_csv"] = []
        for s in subjects[n_samples_for_training:]:
            samples = samples_by_subject[s]
            if len(samples) < self.samples_for_enrollment:
                raise ValueError(
                    f"Not enough samples ({len(samples)}) for enrollment for the subject {s}"
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
                group_by_subject=False,
                references=subjects[n_samples_for_training:],
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


def group_samples_by_subject(samples):

    # Grouping sample sets
    samples_by_subject = dict()
    for s in samples:
        if s.subject not in samples_by_subject:
            samples_by_subject[s.subject] = []
        samples_by_subject[s.subject].append(s)
    return samples_by_subject
