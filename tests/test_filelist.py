#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""A few checks at the Verification Filelist database.
"""

import os

from typing import Any

import numpy as np
import pytest

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import bob.io.base
import bob.io.base.testing_utils

from bob.bio.base.algorithm import Distance
from bob.bio.base.config.dummy.database import database as ATNT_DATABASE
from bob.bio.base.database import (
    AnnotationsLoader,
    CSVDatabase,
    FileListBioDatabase,
    FileSampleLoader,
)
from bob.bio.base.pipelines.pipelines import PipelineSimple
from bob.pipelines import Sample, SampleSet, wrap

legacy_example_dir = os.path.realpath(
    bob.io.base.testing_utils.datafile(".", __name__, "data/")
)

legacy2_example_dir = os.path.realpath(
    bob.io.base.testing_utils.datafile(".", __name__, "data/")
)


example_protocol_path = os.path.realpath(
    bob.io.base.testing_utils.datafile(
        ".", __name__, "data/example_csv_filelist"
    )
)
example_protocol_archive = os.path.realpath(
    bob.io.base.testing_utils.datafile(
        ".", __name__, "data/example_csv_filelist.tar.gz"
    )
)
atnt_protocol_path = os.path.realpath(
    bob.io.base.testing_utils.datafile(".", __name__, "data/atnt")
)


def all_sample(objects: list[Any]) -> bool:
    """Checks that all elements of ``objects`` are valid :class:`Sample` instances."""
    return all(isinstance(o, Sample) for o in objects)


def all_sample_set(objects: list[Any]) -> bool:
    """Checks that all elements of ``objects`` are :class:`SampleSet` instances.

    Also checks that all samples in these samplesets are :class:`Sample` instances.
    """

    return all(isinstance(o, SampleSet) and all_sample(o) for o in objects)


def test_csv_file_list_dev_no_metadata():
    dataset = CSVDatabase(
        name="example_csv_filelist",
        dataset_protocols_path=example_protocol_path,
        protocol="protocol_only_dev",
    )
    assert len(dataset.background_model_samples()) == 8
    assert all_sample(dataset.background_model_samples())

    assert len(dataset.references()) == 2
    assert all_sample_set(dataset.references())
    assert all(len(r) == 4 for r in dataset.references())

    assert len(dataset.probes()) == 2
    assert all_sample_set(dataset.probes())
    assert len(dataset.probes()[0]) == 4
    assert len(dataset.probes()[1]) == 6


def test_csv_file_list_dev_metadata():
    dataset = CSVDatabase(
        name="example_csv_filelist",
        dataset_protocols_path=example_protocol_path,
        protocol="protocol_only_dev_metadata",
    )
    assert len(dataset.background_model_samples()) == 8

    assert all_sample(dataset.background_model_samples())
    assert all(
        hasattr(s, "metadata_1") for s in dataset.background_model_samples()
    )
    assert all(
        hasattr(s, "metadata_2") for s in dataset.background_model_samples()
    )

    assert len(dataset.references()) == 2
    assert all_sample_set(dataset.references())
    assert all(len(r) == 4 for r in dataset.references())
    flat_references = (s for ss in dataset.references() for s in ss)
    assert all(hasattr(s, "metadata_1") for s in flat_references)
    assert all(hasattr(s, "metadata_2") for s in flat_references)

    assert len(dataset.probes()) == 2
    assert all_sample_set(dataset.probes())
    assert all(hasattr(s, "references") for s in dataset.probes())
    assert len(dataset.probes()[0]) == 4
    assert len(dataset.probes()[1]) == 6
    flat_probes = (s for ss in dataset.probes() for s in ss)
    assert all(hasattr(s, "metadata_1") for s in flat_probes)
    assert all(hasattr(s, "metadata_2") for s in flat_probes)


def test_csv_file_list_dev_eval_all_vs_all():
    annotation_directory = os.path.realpath(
        bob.io.base.testing_utils.datafile(
            ".", __name__, "data/example_csv_filelist/annotations"
        )
    )

    def run(filename):
        dataset = CSVDatabase(
            name="example_csv_filelist",
            dataset_protocols_path=filename,
            protocol="protocol_dev_eval",
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory="non-existent",
                    extension=".nothing",
                ),
                AnnotationsLoader(
                    annotation_directory=annotation_directory,
                    annotation_extension=".json",
                    annotation_type="json",
                ),
            ),
        )

        assert dataset.score_all_vs_all
        assert len(dataset.background_model_samples()) == 8
        assert all_sample(dataset.background_model_samples())

        assert len(dataset.references()) == 2
        assert all_sample_set(dataset.references())
        assert len(dataset.references()[0]) == 4
        assert len(dataset.references()[1]) == 4

        assert len(dataset.probes()) == 2
        assert all_sample_set(dataset.probes())
        assert len(dataset.probes()[0]) == 4
        assert len(dataset.probes()[1]) == 4

        eval_references = dataset.references(group="eval")
        assert len(eval_references) == 6
        assert all_sample_set(eval_references)
        assert all(
            len(eval_references[i]) == n
            for i, n in enumerate((2, 2, 3, 1, 1, 1))
        )

        eval_probes = dataset.probes(group="eval")
        assert len(eval_probes) == 8
        assert all_sample_set(eval_probes)
        assert all(
            len(eval_probes[i]) == n
            for i, n in enumerate((2, 2, 3, 1, 1, 2, 1, 1))
        )

        assert len(dataset.all_samples(groups=None)) == 63
        assert all_sample(dataset.all_samples(groups=None))

        # Check the annotations
        for s in dataset.all_samples(groups=None):
            assert isinstance(s.annotations, dict)

        assert len(dataset.template_ids(group="dev")) == 2
        assert len(dataset.template_ids(group="eval")) == 6

        assert len(dataset.groups()) == 3

    run(example_protocol_path)
    run(example_protocol_archive)


def test_csv_file_list_dev_eval_score_norm():
    annotation_directory = os.path.realpath(
        bob.io.base.testing_utils.datafile(
            ".", __name__, "data/example_csv_filelist/annotations"
        )
    )

    def run(filename):
        znorm_dataset = CSVDatabase(
            name="example_csv_filelist",
            dataset_protocols_path=filename,
            protocol="protocol_dev_eval",
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory="non-existent",
                    extension=".nothing",
                ),
                AnnotationsLoader(
                    annotation_directory=annotation_directory,
                    annotation_extension=".json",
                    annotation_type="json",
                ),
            ),
        )

        assert len(znorm_dataset.background_model_samples()) == 8
        assert all_sample(znorm_dataset.background_model_samples())

        assert len(znorm_dataset.references()) == 2
        assert sum(len(s) for s in znorm_dataset.references()) == 8
        assert all_sample_set(znorm_dataset.references())

        assert len(znorm_dataset.probes()) == 2
        assert sum(len(s) for s in znorm_dataset.probes()) == 8
        assert all_sample_set(znorm_dataset.references())

        assert len(znorm_dataset.references(group="eval")) == 6
        assert sum(len(s) for s in znorm_dataset.references(group="eval")) == 10
        assert all_sample_set(znorm_dataset.references(group="eval"))

        assert len(znorm_dataset.probes(group="eval")) == 8
        assert sum(len(s) for s in znorm_dataset.probes(group="eval")) == 13
        assert all_sample_set(znorm_dataset.probes(group="eval"))

        assert len(znorm_dataset.all_samples(groups=None)) == 63
        assert all_sample(znorm_dataset.all_samples(groups=None))

        # Check the annotations
        for s in znorm_dataset.all_samples(groups=None):
            assert isinstance(s.annotations, dict)

        assert len(znorm_dataset.template_ids(group="dev")) == 2
        assert len(znorm_dataset.template_ids(group="eval")) == 6
        assert len(znorm_dataset.groups()) == 3
        assert set(znorm_dataset.groups()) == set(("train", "dev", "eval"))

        # Checking ZT-Norm stuff
        assert len(znorm_dataset.treferences()) == 2
        assert len(znorm_dataset.zprobes()) == 2
        assert sum(len(s) for s in znorm_dataset.zprobes()) == 8

        assert len(znorm_dataset.treferences(proportion=0.5)) == 1
        assert len(znorm_dataset.zprobes(proportion=0.5)) == 1
        assert sum(len(s) for s in znorm_dataset.zprobes(proportion=0.5)) == 4

    run(example_protocol_path)
    run(example_protocol_archive)


def test_csv_file_list_dev_eval_sparse():
    annotation_directory = os.path.realpath(
        bob.io.base.testing_utils.datafile(
            ".", __name__, "data/example_csv_filelist/annotations"
        )
    )

    dataset = CSVDatabase(
        name="example_csv_filelist",
        dataset_protocols_path=example_protocol_path,
        protocol="protocol_dev_eval_sparse",
        transformer=make_pipeline(
            FileSampleLoader(
                data_loader=bob.io.base.load,
                dataset_original_directory="non-existent",
                extension=".nothing",
            ),
            AnnotationsLoader(
                annotation_directory=annotation_directory,
                annotation_extension=".json",
                annotation_type="json",
            ),
        ),
    )

    assert not dataset.score_all_vs_all
    assert len(dataset.background_model_samples()) == 8
    assert all_sample(dataset.background_model_samples())

    assert len(dataset.references()) == 2
    assert all_sample_set(dataset.references())

    probes = dataset.probes()
    assert len(probes) == 2
    assert sum(len(s) for s in probes) == 8

    # Here, 1 comparison per probe
    for p in probes:
        assert len(p.references) == 1
    assert all_sample_set(dataset.references())

    assert len(dataset.references(group="eval")) == 6
    assert all_sample_set(dataset.references(group="eval"))

    probes = dataset.probes(group="eval")
    assert len(probes) == 8
    assert sum(len(s) for s in probes) == 14
    assert all_sample_set(probes)
    # Here, 1 comparison per probe, EXPECT THE FIRST ONE
    for i, p in enumerate(probes):
        if i == 0:
            assert len(p.references) == 2
        else:
            assert len(p.references) == 1

    assert len(dataset.all_samples(groups=None)) == 48
    assert all_sample(dataset.all_samples(groups=None))

    # Check the annotations
    for s in dataset.all_samples(groups=None):
        assert isinstance(s.annotations, dict)

    assert len(dataset.template_ids(group="dev")) == 2
    assert len(dataset.template_ids(group="eval")) == 6

    assert len(dataset.groups()) == 3


def test_csv_file_list_atnt():
    dataset = CSVDatabase(
        name="atnt",
        dataset_protocols_path=atnt_protocol_path,
        protocol="idiap_protocol",
    )
    assert len(dataset.background_model_samples()) == 200
    assert len(dataset.references()) == 20
    assert sum(len(s) for s in dataset.references()) == 100
    assert all(len(s) == 5 for s in dataset.references())
    assert len(dataset.probes()) == 100
    assert sum(len(s) for s in dataset.probes()) == 100
    assert all(len(s) == 1 for s in dataset.probes())
    assert len(dataset.all_samples(groups=["train"])) == 200
    assert len(dataset.all_samples(groups=["dev"])) == 200
    assert len(dataset.all_samples(groups=None)) == 400


def data_loader(path):
    return bob.io.base.load(path)


def run_experiment(dataset):
    def linearize(X):
        X = np.asarray(X)
        return np.reshape(X, (X.shape[0], -1))

    # Testing it in a real recognition systems
    transformer = wrap(
        ["sample"], make_pipeline(FunctionTransformer(linearize))
    )

    pipeline_simple = PipelineSimple(transformer, Distance())

    return pipeline_simple(
        dataset.background_model_samples(),
        dataset.references(),
        dataset.probes(),
    )


def test_atnt_experiment():
    dataset = CSVDatabase(
        name="atnt",
        dataset_protocols_path=atnt_protocol_path,
        protocol="idiap_protocol",
        transformer=FileSampleLoader(
            data_loader=data_loader,
            dataset_original_directory=ATNT_DATABASE.original_directory,
            extension=".pgm",
        ),
    )

    scores = run_experiment(dataset)
    assert len(scores) == 100
    assert all(len(s) == 20 for s in scores)


####
# Testing the Legacy file list
####


def test_query():
    db = FileListBioDatabase(
        os.path.join(legacy_example_dir, "example_filelist"),
        "test",
        use_dense_probe_file_list=False,
    )

    assert (
        len(db.groups()) == 5
    )  # 5 groups (dev, eval, world, optional_world_1, optional_world_2)

    assert len(db.client_ids()) == 6  # 6 client ids for world, dev and eval
    assert len(db.client_ids(groups="world")) == 2  # 2 client ids for world
    assert (
        len(db.client_ids(groups="optional_world_1")) == 2
    )  # 2 client ids for optional world 1
    assert (
        len(db.client_ids(groups="optional_world_2")) == 2
    )  # 2 client ids for optional world 2
    assert len(db.client_ids(groups="dev")) == 2  # 2 client ids for dev
    assert len(db.client_ids(groups="eval")) == 2  # 2 client ids for eval

    assert (
        len(db.tclient_ids()) == 2
    )  # 2 client ids for T-Norm score normalization
    assert (
        len(db.zclient_ids()) == 2
    )  # 2 client ids for Z-Norm score normalization

    assert (
        len(db.model_ids_with_protocol()) == 6
    )  # 6 model ids for world, dev and eval
    assert (
        len(db.model_ids_with_protocol(groups="world")) == 2
    )  # 2 model ids for world
    assert (
        len(db.model_ids_with_protocol(groups="optional_world_1")) == 2
    )  # 2 model ids for optional world 1
    assert (
        len(db.model_ids_with_protocol(groups="optional_world_2")) == 2
    )  # 2 model ids for optional world 2
    assert (
        len(db.model_ids_with_protocol(groups="dev")) == 2
    )  # 2 model ids for dev
    assert (
        len(db.model_ids_with_protocol(groups="eval")) == 2
    )  # 2 model ids for eval

    assert (
        len(db.tmodel_ids_with_protocol()) == 2
    )  # 2 model ids for T-Norm score normalization

    assert len(db.objects(groups="world")) == 8  # 8 samples in the world set

    assert (
        len(db.objects(groups="dev", purposes="enroll")) == 8
    )  # 8 samples for enrollment in the dev set
    assert (
        len(db.objects(groups="dev", purposes="enroll", model_ids="3")) == 4
    )  # 4 samples for to enroll model '3' in the dev set
    assert (
        len(db.objects(groups="dev", purposes="enroll", model_ids="7")) == 0
    )  # 0 samples for enrolling model '7' (it is a T-Norm model)
    assert (
        len(db.objects(groups="dev", purposes="probe")) == 8
    )  # 8 samples as probes in the dev set
    assert (
        len(db.objects(groups="dev", purposes="probe", classes="client")) == 8
    )  # 8 samples as client probes in the dev set
    assert (
        len(db.objects(groups="dev", purposes="probe", classes="impostor")) == 4
    )  # 4 samples as impostor probes in the dev set

    assert (
        len(db.tobjects(groups="dev")) == 8
    )  # 8 samples for enrolling T-norm models
    assert (
        len(db.tobjects(groups="dev", model_ids="7")) == 4
    )  # 4 samples for enrolling T-norm model '7'
    assert (
        len(db.tobjects(groups="dev", model_ids="3")) == 0
    )  # 0 samples for enrolling T-norm model '3' (no T-Norm model)
    assert (
        len(db.zobjects(groups="dev")) == 8
    )  # 8 samples for Z-norm impostor accesses

    assert db.client_id_from_model_id("1", group=None) == "1"
    assert db.client_id_from_model_id("3", group=None) == "3"
    assert db.client_id_from_model_id("6", group=None) == "6"
    assert db.client_id_from_t_model_id("7", group=None) == "7"


def test_query_protocol():
    db = FileListBioDatabase(
        legacy_example_dir,
        "test",
        protocol="example_filelist",
        use_dense_probe_file_list=False,
    )

    assert (
        len(db.groups()) == 5
    )  # 5 groups (dev, eval, world, optional_world_1, optional_world_2)

    assert len(db.client_ids()) == 6  # 6 client ids for world, dev and eval
    assert (
        len(
            db.client_ids(
                groups="world",
            )
        )
        == 2
    )  # 2 client ids for world
    assert (
        len(
            db.client_ids(
                groups="optional_world_1",
            )
        )
        == 2
    )  # 2 client ids for optional world 1
    assert (
        len(
            db.client_ids(
                groups="optional_world_2",
            )
        )
        == 2
    )  # 2 client ids for optional world 2
    assert (
        len(
            db.client_ids(
                groups="dev",
            )
        )
        == 2
    )  # 2 client ids for dev
    assert (
        len(
            db.client_ids(
                groups="eval",
            )
        )
        == 2
    )  # 2 client ids for eval

    assert (
        len(db.tclient_ids()) == 2
    )  # 2 client ids for T-Norm score normalization
    assert (
        len(db.zclient_ids()) == 2
    )  # 2 client ids for Z-Norm score normalization

    assert (
        len(db.model_ids_with_protocol()) == 6
    )  # 6 model ids for world, dev and eval
    assert (
        len(
            db.model_ids_with_protocol(
                groups="world",
            )
        )
        == 2
    )  # 2 model ids for world
    assert (
        len(
            db.model_ids_with_protocol(
                groups="optional_world_1",
            )
        )
        == 2
    )  # 2 model ids for optional world 1
    assert (
        len(
            db.model_ids_with_protocol(
                groups="optional_world_2",
            )
        )
        == 2
    )  # 2 model ids for optional world 2
    assert (
        len(
            db.model_ids_with_protocol(
                groups="dev",
            )
        )
        == 2
    )  # 2 model ids for dev
    assert (
        len(
            db.model_ids_with_protocol(
                groups="eval",
            )
        )
        == 2
    )  # 2 model ids for eval

    assert (
        len(db.tmodel_ids_with_protocol()) == 2
    )  # 2 model ids for T-Norm score normalization

    assert (
        len(
            db.objects(
                groups="world",
            )
        )
        == 8
    )  # 8 samples in the world set

    assert (
        len(
            db.objects(
                groups="dev",
                purposes="enroll",
            )
        )
        == 8
    )  # 8 samples for enrollment in the dev set
    assert (
        len(
            db.objects(
                groups="dev",
                purposes="enroll",
                model_ids="3",
            )
        )
        == 4
    )  # 4 samples for to enroll model '3' in the dev set
    assert (
        len(
            db.objects(
                groups="dev",
                purposes="enroll",
                model_ids="7",
            )
        )
        == 0
    )  # 0 samples for enrolling model '7' (it is a T-Norm model)
    assert (
        len(
            db.objects(
                groups="dev",
                purposes="probe",
            )
        )
        == 8
    )  # 8 samples as probes in the dev set
    assert (
        len(
            db.objects(
                groups="dev",
                purposes="probe",
                classes="client",
            )
        )
        == 8
    )  # 8 samples as client probes in the dev set
    assert (
        len(
            db.objects(
                groups="dev",
                purposes="probe",
                classes="impostor",
            )
        )
        == 4
    )  # 4 samples as impostor probes in the dev set

    assert (
        len(
            db.tobjects(
                groups="dev",
            )
        )
        == 8
    )  # 8 samples for enrolling T-norm models
    assert (
        len(
            db.tobjects(
                groups="dev",
                model_ids="7",
            )
        )
        == 4
    )  # 4 samples for enrolling T-norm model '7'
    assert (
        len(
            db.tobjects(
                groups="dev",
                model_ids="3",
            )
        )
        == 0
    )  # 0 samples for enrolling T-norm model '3' (no T-Norm model)
    assert (
        len(db.zobjects(groups="dev")) == 8
    )  # 8 samples for Z-norm impostor accesses

    assert db.client_id_from_model_id("1", group=None) == "1"
    assert db.client_id_from_model_id("3", group=None) == "3"
    assert db.client_id_from_model_id("6", group=None) == "6"
    assert db.client_id_from_t_model_id("7", group=None) == "7"

    # check other protocols
    assert len(db.objects(protocol="non-existent")) == 0

    prot = "example_filelist2"
    assert (
        len(db.model_ids_with_protocol(protocol=prot)) == 3
    )  # 3 model ids for dev only
    with pytest.raises(ValueError):
        # eval does not exist for this protocol
        db.model_ids_with_protocol(protocol=prot, groups="eval")
    assert len(db.objects(protocol=prot, groups="dev", purposes="enroll")) == 12
    assert len(db.objects(protocol=prot, groups="dev", purposes="probe")) == 9


def test_noztnorm():
    db = FileListBioDatabase(
        os.path.join(legacy_example_dir, "example_filelist2"), "test"
    )
    assert len(db.all_files())


def test_query_dense():
    db = FileListBioDatabase(
        os.path.join(legacy_example_dir, "example_filelist"),
        "test",
        use_dense_probe_file_list=True,
    )

    assert len(db.objects(groups="world")) == 8  # 8 samples in the world set

    assert (
        len(db.objects(groups="dev", purposes="enroll")) == 8
    )  # 8 samples for enrollment in the dev set
    assert (
        len(db.objects(groups="dev", purposes="probe")) == 8
    )  # 8 samples as probes in the dev set


def test_annotation():
    db = FileListBioDatabase(
        os.path.join(legacy_example_dir, "example_filelist"),
        "test",
        use_dense_probe_file_list=False,
        annotation_directory=os.path.join(
            legacy_example_dir, "example_filelist"
        ),
        annotation_type="json",
    )
    f = [o for o in db.objects() if o.path == "data/model4_session1_sample2"][0]

    annots = db.annotations(f)

    assert annots is not None
    assert "reye" in annots
    assert "leye" in annots
    assert annots["reye"] == [20, 10]
    assert annots["leye"] == [40, 30]


def test_multiple_extensions():
    # check that the old behavior still works
    db = FileListBioDatabase(
        legacy_example_dir,
        "test",
        protocol="example_filelist",
        use_dense_probe_file_list=False,
        original_directory=os.path.join(legacy_example_dir, "example_filelist"),
        original_extension=".json",
    )

    file = bob.bio.base.database.BioFile(
        4, "data/model4_session1_sample2", "data/model4_session1_sample2"
    )

    file_name = db.original_file_name(file, True)
    assert file_name == os.path.join(
        legacy_example_dir, "example_filelist", file.path + ".json"
    )

    # check that the new behavior works as well
    db = FileListBioDatabase(
        legacy_example_dir,
        "test",
        protocol="example_filelist",
        use_dense_probe_file_list=False,
        original_directory=os.path.join(legacy_example_dir, "example_filelist"),
        original_extension=[".jpg", ".json"],
    )
    file_name = db.original_file_name(file)
    assert file_name == os.path.join(
        legacy_example_dir, "example_filelist", file.path + ".json"
    )

    file = bob.bio.base.database.BioFile(
        4, "data/model4_session1_sample1", "data/model4_session1_sample1"
    )
    with pytest.raises(IOError):
        db.original_file_name(file, False)
