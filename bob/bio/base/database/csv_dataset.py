#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import os
from bob.pipelines import Sample, DelayedSample, SampleSet
import csv
import bob.io.base
import functools
from abc import ABCMeta, abstractmethod


class CSVSampleLoaderAbstract(metaclass=ABCMeta):
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

    def __init__(self, data_loader, extension=""):
        self.data_loader = data_loader
        self.extension = extension
        self.excluding_attributes = ["_data", "load", "key"]

    @abstractmethod
    def __call__(self, filename):
        pass

    @abstractmethod
    def convert_row_to_sample(self, row, header):
        pass

    @abstractmethod
    def convert_samples_to_samplesets(self, samples, group_by_subject=True):
        pass


class CSVToSampleLoader(CSVSampleLoaderAbstract):
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
        kwargs = dict([[h, r] for h,r in zip(header[2:], row[2:])])
        return DelayedSample(
            functools.partial(self.data_loader, os.path.join(path, self.extension)),
            key=path,
            subject=subject,
            **kwargs,
        )

    def convert_samples_to_samplesets(self, samples, group_by_subject=True):
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
                sample_sets[s.subject].append(s)
            return list(sample_sets.values())

        else:
            return [SampleSet([s], **get_attribute_from_sample(s)) for s in samples]


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

        protocol: str
          The name of the protocol

        csv_to_sample_loader:


    """

    def __init__(
        self,
        dataset_path,
        protocol,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=bob.io.base.load, extension=""
        ),
    ):
        def get_paths():

            if not os.path.exists(dataset_path):
                raise ValueError(f"The path `{dataset_path}` was not found")

            # TODO: Unzip file if dataset path is a zip
            protocol_path = os.path.join(dataset_path, protocol)
            if not os.path.exists(protocol_path):
                raise ValueError(f"The protocol `{protocol}` was not found")

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

        if purpose == "enroll":
            cache_label = "dev_enroll_csv" if group == "dev" else "eval_enroll_csv"
        else:
            cache_label = "dev_probe_csv" if group == "dev" else "eval_probe_csv"

        if self.cache[cache_label] is not None:
            return self.cache[cache_label]

        probes_data = self.csv_to_sample_loader(self.__dict__[cache_label])

        sample_sets = self.csv_to_sample_loader.convert_samples_to_samplesets(
            probes_data, group_by_subject=group_by_subject
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
