#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import csv
import functools
import itertools
import logging
import os

from typing import List, Union

import numpy as np

import bob.io.base

from bob.bio.base.pipelines.abstract_classes import Database
from bob.extension.download import list_dir, search_file
from bob.pipelines import DelayedSample, SampleSet
from bob.pipelines.sample_loaders import CSVToSampleLoader

from .legacy import check_parameters_for_validity

logger = logging.getLogger(__name__)


def convert_samples_to_samplesets(
    samples, group_by_reference_id=True, references=None
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
            else [
                SampleSet([s], parent=s, references=references) for s in samples
            ]
        )


class LSTToSampleLoader(CSVToSampleLoader):
    """
    Simple mechanism that converts the lines of a LST file to
    :any:`bob.pipelines.DelayedSample` or :any:`bob.pipelines.SampleSet`
    """

    def transform(self, X):
        X.seek(0)
        reader = csv.reader(X, delimiter=" ")
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
                kwargs = {"subject_id": str(subject)}

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(
                    self.dataset_original_directory, path + self.extension
                ),
            ),
            key=path,
            reference_id=reference_id,
            **kwargs,
        )


class CSVToSampleLoaderBiometrics(CSVToSampleLoader):
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
            A python function that can be called parameterlessly, to load the
            sample in question from whatever medium

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
        reference_id_equal_subject_id=True,
    ):
        super().__init__(
            data_loader=data_loader,
            extension=extension,
            dataset_original_directory=dataset_original_directory,
        )
        self.reference_id_equal_subject_id = reference_id_equal_subject_id

    def convert_row_to_sample(self, row, header):
        path = row[0]
        reference_id = row[1]

        kwargs = dict(
            [[str(h).lower(), r] for h, r in zip(header[2:], row[2:])]
        )
        if self.reference_id_equal_subject_id:
            kwargs["subject_id"] = reference_id
        else:
            if "subject_id" not in kwargs:
                raise ValueError(f"`subject_id` not available in {header}")

        return DelayedSample(
            functools.partial(
                self.data_loader,
                os.path.join(
                    self.dataset_original_directory, path + self.extension
                ),
            ),
            key=path,
            reference_id=reference_id,
            **kwargs,
        )


class CSVDataset(Database):
    """
    Generic filelist dataset for :any:` bob.bio.base.pipelines.PipelineSimple` pipeline.
    Check :any:`pipeline_simple_features` for more details about the PipelineSimple Dataset
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
    for enrollment (:any:`bob.bio.base.pipelines.Database.references`) and
    probing (:any:`bob.bio.base.pipelines.Database.probes`).
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
    contains data for training (`bob.bio.base.pipelines.Database.background_model_samples`)

    Parameters
    ----------

        name: str
            Name of the dataset (root folder name containing the protocol folders)

        protocol: str
            Name of the protocol (folder name containing the dev, eval and norm folders)

        dataset_protocol_path: str
          Absolute path or a tarball of the dataset protocol description.

        protocol_name: str
          The name of the protocol

        csv_to_sample_loader: `bob.pipelines.sample_loaders.CSVToSampleLoader`
            Base class that whose objective is to generate :any:`bob.pipelines.Sample`
            and/or :any:`bob.pipelines.SampleSet` from csv rows

        is_sparse: bool
            If True, will look for a `for_scores.lst` file instead of a `for_probes.lst`
            (legacy format)

        score_all_vs_all: bool
            Optimization trick for Dask. If True, all references will be passed for
            scoring against the probes.

        group_probes_by_reference_id: bool
            If True, probe SampleSet will contain all the samples with a given
            `reference_id`, otherwise, one SampleSet will be created for each sample.


    """

    def __init__(
        self,
        *,
        name,
        protocol,
        dataset_protocol_path,
        csv_to_sample_loader=None,
        is_sparse=False,
        score_all_vs_all=False,
        group_probes_by_reference_id=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            protocol=protocol,
            score_all_vs_all=score_all_vs_all,
            **kwargs,
        )
        self.dataset_protocol_path = dataset_protocol_path
        self.is_sparse = is_sparse
        self.group_probes_by_reference_id = group_probes_by_reference_id
        if csv_to_sample_loader is None:
            csv_to_sample_loader = CSVToSampleLoaderBiometrics(
                data_loader=bob.io.base.load,
                dataset_original_directory="",
                extension="",
            )

        def get_paths():

            if not os.path.exists(dataset_protocol_path):
                raise ValueError(
                    f"The path `{dataset_protocol_path}` was not found"
                )

            # Here we are handling the legacy
            train_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(name, protocol, "norm", "train_world.lst"),
                    os.path.join(name, protocol, "norm", "train_world.csv"),
                ],
            )

            dev_enroll_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(name, protocol, "dev", "for_models.lst"),
                    os.path.join(name, protocol, "dev", "for_models.csv"),
                ],
            )

            legacy_probe = (
                "for_scores.lst" if self.is_sparse else "for_probes.lst"
            )
            dev_probe_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(name, protocol, "dev", legacy_probe),
                    os.path.join(name, protocol, "dev", "for_probes.csv"),
                ],
            )

            eval_enroll_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(name, protocol, "eval", "for_models.lst"),
                    os.path.join(name, protocol, "eval", "for_models.csv"),
                ],
            )

            eval_probe_csv = search_file(
                dataset_protocol_path,
                [
                    os.path.join(name, protocol, "eval", legacy_probe),
                    os.path.join(name, protocol, "eval", "for_probes.csv"),
                ],
            )

            # The minimum required is to have `dev_enroll_csv` and `dev_probe_csv`

            # Dev
            if dev_enroll_csv is None:
                raise ValueError(
                    f"The file `{dev_enroll_csv}` is required and it was not found"
                )

            if dev_probe_csv is None:
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
            self.csv_to_sample_loader.transform(self.train_csv)
            if self.cache["train"] is None
            else self.cache["train"]
        )

        return self.cache["train"]

    def _get_samplesets(
        self,
        group="dev",
        cache_key=None,
        group_by_reference_id=False,
        fetching_probes=False,
        is_sparse=False,
    ):

        if self.cache[cache_key] is not None:
            return self.cache[cache_key]

        # Getting samples from CSV
        samples = self.csv_to_sample_loader.transform(
            self.__getattribute__(cache_key)
        )

        references = None
        if fetching_probes and is_sparse:

            # Checking if `is_sparse` was set properly
            if len(samples) > 0 and not hasattr(
                samples[0], "compare_reference_id"
            ):
                ValueError(
                    f"Attribute `compare_reference_id` not found in `{samples[0]}`."
                    "Make sure this attribute exists in your dataset if `is_sparse=True`"
                )

            sparse_samples = dict()
            for s in samples:
                if s.key in sparse_samples:
                    sparse_samples[s.key].references.append(
                        s.compare_reference_id
                    )
                else:
                    s.references = [s.compare_reference_id]
                    sparse_samples[s.key] = s
            samples = sparse_samples.values()
        else:
            if fetching_probes:
                references = list(
                    set([s.reference_id for s in self.references(group=group)])
                )

        sample_sets = convert_samples_to_samplesets(
            samples,
            group_by_reference_id=group_by_reference_id,
            references=references,
        )

        self.cache[cache_key] = sample_sets

        return self.cache[cache_key]

    def references(self, group="dev"):
        cache_key = "dev_enroll_csv" if group == "dev" else "eval_enroll_csv"

        return self._get_samplesets(
            group=group, cache_key=cache_key, group_by_reference_id=True
        )

    def probes(self, group="dev"):
        cache_key = "dev_probe_csv" if group == "dev" else "eval_probe_csv"

        return self._get_samplesets(
            group=group,
            cache_key=cache_key,
            group_by_reference_id=self.group_probes_by_reference_id,
            fetching_probes=True,
            is_sparse=self.is_sparse,
        )

    def all_samples(
        self, groups: Union[List[str], str, None] = None
    ) -> List[bob.pipelines.Sample]:
        """
        Reads and returns all the samples in ``groups``.

        Parameters
        ----------
        groups:
            Groups to consider ('train', 'dev', and/or 'eval'). If ``None`` is
            given, returns the samples from all existing groups.

        Returns
        -------
        samples:
            List of :class:`bob.pipelines.Sample` objects.
        """
        valid_groups = []
        if self.train_csv:
            valid_groups.append("train")
        if self.dev_enroll_csv and self.dev_probe_csv:
            valid_groups.append("dev")
        if self.eval_enroll_csv and self.eval_probe_csv:
            valid_groups.append("eval")
        groups = list(
            check_parameters_for_validity(
                parameters=groups,
                parameter_description="groups",
                valid_parameters=valid_groups,
                default_parameters=valid_groups,
            )
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
                samples = samples + self.csv_to_sample_loader.transform(
                    self.__getattribute__(label)
                )
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

    def protocols(self):
        return list_dir(self.dataset_protocol_path, folders=True, files=False)


class CSVDatasetZTNorm(CSVDataset):
    """
    Generic filelist dataset for :any:`bob.bio.base.pipelines.PipelineSimple` pipelines.
    Check :any:`pipeline_simple_features` for more details about the PipelineSimple Dataset
    interface.

    This dataset interface takes as in put a :any:`CSVDataset` as input and have two extra methods:
    :any:`CSVDatasetZTNorm.zprobes` and :any:`CSVDatasetZTNorm.treferences`.

    To create a new dataset, you need to provide a directory structure similar to the one below:

    .. code-block:: text

       my_dataset/
       my_dataset/my_protocol/norm/train_world.csv
       my_dataset/my_protocol/norm/for_znorm.csv
       my_dataset/my_protocol/norm/for_tnorm.csv
       my_dataset/my_protocol/dev/for_models.csv
       my_dataset/my_protocol/dev/for_probes.csv
       my_dataset/my_protocol/eval/for_models.csv
       my_dataset/my_protocol/eval/for_probes.csv

    Parameters
    ----------

      database: :any:`CSVDataset`
         :any:`CSVDataset` to be aggregated

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # create_cache
        self.cache["znorm_csv_dev"] = None
        self.cache["znorm_csv_eval"] = None
        self.cache["tnorm_csv"] = None

        znorm_csv = search_file(
            self.dataset_protocol_path,
            [
                os.path.join(self.protocol, "norm", "for_znorm.lst"),
                os.path.join(self.protocol, "norm", "for_znorm.csv"),
            ],
        )

        tnorm_csv = search_file(
            self.dataset_protocol_path,
            [
                os.path.join(self.protocol, "norm", "for_tnorm.lst"),
                os.path.join(self.protocol, "norm", "for_tnorm.csv"),
            ],
        )

        if znorm_csv is None:
            raise ValueError(
                f"The file `for_znorm.lst` is required and it was not found in `{self.protocol}/norm` "
            )

        if tnorm_csv is None:
            raise ValueError(
                f"The file `for_tnorm.csv` is required and it was not found `{self.protocol}/norm`"
            )

        self.znorm_csv_dev = znorm_csv
        self.znorm_csv_eval = znorm_csv
        self.tnorm_csv = tnorm_csv

    def zprobes(self, group="dev", proportion=1.0):

        if proportion <= 0 or proportion > 1:
            raise ValueError(
                f"Invalid proportion value ({proportion}). Values allowed from [0-1]"
            )

        cache_key = f"znorm_csv_{group}"
        samplesets = self._get_samplesets(
            group=group,
            cache_key=cache_key,
            group_by_reference_id=self.group_probes_by_reference_id,
            fetching_probes=True,
            is_sparse=False,
        )

        zprobes = samplesets[: int(len(samplesets) * proportion)]

        return zprobes

    def treferences(self, covariate="sex", proportion=1.0):

        if proportion <= 0 or proportion > 1:
            raise ValueError(
                f"Invalid proportion value ({proportion}). Values allowed from [0-1]"
            )

        cache_key = "tnorm_csv"
        samplesets = self._get_samplesets(
            group="dev",
            cache_key=cache_key,
            group_by_reference_id=True,
        )

        treferences = samplesets[: int(len(samplesets) * proportion)]

        return treferences


class CSVDatasetCrossValidation(Database):
    """
    Generic filelist dataset for :any:`bob.bio.base.pipelines.PipelineSimple` pipeline that
    handles **CROSS VALIDATION**.

    Check :any:`pipeline_simple_features` for more details about the PipelineSimple Dataset
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

    csv_to_sample_loader: `bob.pipelines.sample_loaders.CSVToSampleLoader`
        Base class that whose objective is to generate :any:`bob.pipelines.Sample`
        and/or :any:`bob.pipelines.SampleSet` from csv rows

    """

    def __init__(
        self,
        *,
        name,
        protocol="Default",
        csv_file_name="metadata.csv",
        random_state=0,
        test_size=0.8,
        samples_for_enrollment=1,
        csv_to_sample_loader=None,
        score_all_vs_all=True,
        group_probes_by_reference_id=False,
        **kwargs,
    ):
        super().__init__(
            name=name,
            protocol=protocol,
            score_all_vs_all=score_all_vs_all,
            **kwargs,
        )
        if csv_to_sample_loader is None:
            csv_to_sample_loader = CSVToSampleLoaderBiometrics(
                data_loader=bob.io.base.load,
                dataset_original_directory="",
                extension="",
            )

        def get_dict_cache():
            cache = dict()
            cache["train"] = None
            cache["dev_enroll_csv"] = None
            cache["dev_probe_csv"] = None
            return cache

        self.random_state = random_state
        self.cache = get_dict_cache()
        self.csv_to_sample_loader = csv_to_sample_loader
        self.csv_file_name = open(csv_file_name)
        self.samples_for_enrollment = samples_for_enrollment
        self.test_size = test_size
        self.group_probes_by_reference_id = group_probes_by_reference_id

        if self.test_size < 0 and self.test_size > 1:
            raise ValueError(
                f"`test_size` should be between 0 and 1. {test_size} is provided"
            )

    def _do_cross_validation(self):

        # Shuffling samples by reference_id
        samples_by_reference_id = group_samples_by_reference_id(
            self.csv_to_sample_loader.transform(self.csv_file_name)
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
                convert_samples_to_samplesets(
                    samples[0 : self.samples_for_enrollment]
                )[0]
            )

            self.cache["dev_probe_csv"] += convert_samples_to_samplesets(
                samples[self.samples_for_enrollment :],
                group_by_reference_id=self.group_probes_by_reference_id,
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
            samples = samples + [
                s for s_set in self.references(group) for s in s_set
            ]
            samples = samples + [
                s for s_set in self.probes(group) for s in s_set
            ]
        return samples

    def groups(self):
        return list(("train", "dev"))

    def protocols(self):
        return list((self.protocol,))


def group_samples_by_reference_id(samples):

    # Grouping sample sets
    samples_by_reference_id = dict()
    for s in samples:
        if s.reference_id not in samples_by_reference_id:
            samples_by_reference_id[s.reference_id] = []
        samples_by_reference_id[s.reference_id].append(s)
    return samples_by_reference_id
