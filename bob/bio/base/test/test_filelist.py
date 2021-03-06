#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""A few checks at the Verification Filelist database.
"""

import os
import bob.io.base
import bob.io.base.test_utils
from bob.bio.base.database import (
    CSVDataset,
    CSVDatasetCrossValidation,
    LSTToSampleLoader,
    CSVDatasetZTNorm,
)
import nose.tools
from bob.pipelines import DelayedSample, SampleSet
import numpy as np
from bob.bio.base.test.utils import atnt_database_directory
from bob.bio.base.pipelines.vanilla_biometrics import (
    Distance,
    VanillaBiometricsPipeline,
)
from bob.bio.base.database import FileListBioDatabase
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
from bob.pipelines.datasets import AnnotationsLoader, CSVToSampleLoader


legacy_example_dir = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/example_filelist")
)

legacy2_example_dir = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/example_filelist2")
)


example_dir = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/example_csv_filelist")
)
atnt_protocol_path = os.path.realpath(
    bob.io.base.test_utils.datafile(".", __name__, "data/atnt")
)

atnt_protocol_path_cross_validation = os.path.join(
    os.path.realpath(
        bob.io.base.test_utils.datafile(".", __name__, "data/atnt/cross_validation/")
    ),
    "metadata.csv",
)


def check_all_true(list_of_something, something):
    """
    Assert if list of `Something` contains `Something`
    """
    return np.alltrue([isinstance(s, something) for s in list_of_something])


def test_csv_file_list_dev_only():

    dataset = CSVDataset(example_dir, "protocol_only_dev")
    assert len(dataset.background_model_samples()) == 8
    assert check_all_true(dataset.background_model_samples(), DelayedSample)

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.probes(), SampleSet)


def test_csv_file_list_dev_only_metadata():

    dataset = CSVDataset(example_dir, "protocol_only_dev_metadata")
    assert len(dataset.background_model_samples()) == 8

    assert check_all_true(dataset.background_model_samples(), DelayedSample)
    assert np.alltrue(
        ["metadata_1" in s.__dict__ for s in dataset.background_model_samples()]
    )
    assert np.alltrue(
        ["metadata_2" in s.__dict__ for s in dataset.background_model_samples()]
    )

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)
    assert np.alltrue(["metadata_1" in s.__dict__ for s in dataset.references()])
    assert np.alltrue(["metadata_2" in s.__dict__ for s in dataset.references()])

    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.probes(), SampleSet)
    assert np.alltrue(["metadata_1" in s.__dict__ for s in dataset.probes()])
    assert np.alltrue(["metadata_2" in s.__dict__ for s in dataset.probes()])
    assert np.alltrue(["references" in s.__dict__ for s in dataset.probes()])


def test_csv_file_list_dev_eval():

    annotation_directory = os.path.realpath(
        bob.io.base.test_utils.datafile(
            ".", __name__, "data/example_csv_filelist/annotations"
        )
    )

    def run(filename):
        dataset = CSVDataset(
            filename,
            "protocol_dev_eval",
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory="",
                    extension="",
                ),
                AnnotationsLoader(
                    annotation_directory=annotation_directory,
                    annotation_extension=".pos",
                    annotation_type="eyecenter",
                ),
            ),
        )
        assert len(dataset.background_model_samples()) == 8
        assert check_all_true(dataset.background_model_samples(), DelayedSample)

        assert len(dataset.references()) == 2
        assert check_all_true(dataset.references(), SampleSet)

        assert len(dataset.probes()) == 8
        assert check_all_true(dataset.references(), SampleSet)

        assert len(dataset.references(group="eval")) == 6
        assert check_all_true(dataset.references(group="eval"), SampleSet)

        assert len(dataset.probes(group="eval")) == 13
        assert check_all_true(dataset.probes(group="eval"), SampleSet)

        assert len(dataset.all_samples(groups=None)) == 47
        assert check_all_true(dataset.all_samples(groups=None), DelayedSample)

        # Check the annotations
        for s in dataset.all_samples(groups=None):
            assert isinstance(s.annotations, dict)

        assert len(dataset.reference_ids(group="dev")) == 2
        assert len(dataset.reference_ids(group="eval")) == 6

        assert len(dataset.groups()) == 3

    run(example_dir)
    run(example_dir + ".tar.gz")


def test_csv_file_list_dev_eval_score_norm():

    annotation_directory = os.path.realpath(
        bob.io.base.test_utils.datafile(
            ".", __name__, "data/example_csv_filelist/annotations"
        )
    )

    def run(filename):
        dataset = CSVDataset(
            filename,
            "protocol_dev_eval",
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory="",
                    extension="",
                ),
                AnnotationsLoader(
                    annotation_directory=annotation_directory,
                    annotation_extension=".pos",
                    annotation_type="eyecenter",
                ),
            ),
        )

        znorm_dataset = CSVDatasetZTNorm(dataset)

        assert len(znorm_dataset.background_model_samples()) == 8
        assert check_all_true(znorm_dataset.background_model_samples(), DelayedSample)

        assert len(znorm_dataset.references()) == 2
        assert check_all_true(znorm_dataset.references(), SampleSet)

        assert len(znorm_dataset.probes()) == 8
        assert check_all_true(znorm_dataset.references(), SampleSet)

        assert len(znorm_dataset.references(group="eval")) == 6
        assert check_all_true(znorm_dataset.references(group="eval"), SampleSet)

        assert len(znorm_dataset.probes(group="eval")) == 13
        assert check_all_true(znorm_dataset.probes(group="eval"), SampleSet)

        assert len(znorm_dataset.all_samples(groups=None)) == 47
        assert check_all_true(znorm_dataset.all_samples(groups=None), DelayedSample)

        # Check the annotations
        for s in znorm_dataset.all_samples(groups=None):
            assert isinstance(s.annotations, dict)

        assert len(znorm_dataset.reference_ids(group="dev")) == 2
        assert len(znorm_dataset.reference_ids(group="eval")) == 6
        assert len(znorm_dataset.groups()) == 3

        ## Checking ZT-Norm stuff
        assert len(znorm_dataset.treferences()) == 2
        assert len(znorm_dataset.zprobes()) == 8

        assert len(znorm_dataset.treferences(proportion=0.5)) == 1
        assert len(znorm_dataset.zprobes(proportion=0.5)) == 4

    run(example_dir)
    run(example_dir + ".tar.gz")


def test_csv_file_list_dev_eval_sparse():

    annotation_directory = os.path.realpath(
        bob.io.base.test_utils.datafile(
            ".", __name__, "data/example_csv_filelist/annotations"
        )
    )

    dataset = CSVDataset(
        example_dir,
        "protocol_dev_eval_sparse",
        csv_to_sample_loader=make_pipeline(
            CSVToSampleLoader(
                data_loader=bob.io.base.load,
                dataset_original_directory="",
                extension="",
            ),
            AnnotationsLoader(
                annotation_directory=annotation_directory,
                annotation_extension=".pos",
                annotation_type="eyecenter",
            ),
        ),
        is_sparse=True,
    )

    assert len(dataset.background_model_samples()) == 8
    assert check_all_true(dataset.background_model_samples(), DelayedSample)

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)

    probes = dataset.probes()
    assert len(probes) == 8

    # here, 1 comparisons comparison per probe
    for p in probes:
        assert len(p.references) == 1
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.references(group="eval")) == 6
    assert check_all_true(dataset.references(group="eval"), SampleSet)

    probes = dataset.probes(group="eval")
    assert len(probes) == 13
    assert check_all_true(probes, SampleSet)
    # Here, 1 comparison per probe, EXPECT THE FIRST ONE
    for i, p in enumerate(probes):
        if i == 0:
            assert len(p.references) == 2
        else:
            assert len(p.references) == 1

    assert len(dataset.all_samples(groups=None)) == 48
    assert check_all_true(dataset.all_samples(groups=None), DelayedSample)

    # Check the annotations
    for s in dataset.all_samples(groups=None):
        assert isinstance(s.annotations, dict)

    assert len(dataset.reference_ids(group="dev")) == 2
    assert len(dataset.reference_ids(group="eval")) == 6

    assert len(dataset.groups()) == 3


def test_lst_file_list_dev_eval():

    dataset = CSVDataset(
        legacy_example_dir,
        "",
        csv_to_sample_loader=LSTToSampleLoader(
            data_loader=bob.io.base.load, dataset_original_directory="", extension="",
        ),
    )

    assert len(dataset.background_model_samples()) == 8

    assert check_all_true(dataset.background_model_samples(), DelayedSample)

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.probes()) == 10
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.references(group="eval")) == 2
    assert check_all_true(dataset.references(group="eval"), SampleSet)

    assert len(dataset.probes(group="eval")) == 8
    assert check_all_true(dataset.probes(group="eval"), SampleSet)

    assert len(dataset.all_samples(groups=None)) == 42
    assert check_all_true(dataset.all_samples(groups=None), DelayedSample)

    assert len(dataset.reference_ids(group="dev")) == 2
    assert len(dataset.reference_ids(group="eval")) == 2

    assert len(dataset.groups()) == 3


def test_lst_file_list_dev_eval_sparse():

    dataset = CSVDataset(
        legacy_example_dir,
        "",
        csv_to_sample_loader=LSTToSampleLoader(
            data_loader=bob.io.base.load, dataset_original_directory="", extension="",
        ),
        is_sparse=True,
    )

    assert len(dataset.background_model_samples()) == 8

    assert check_all_true(dataset.background_model_samples(), DelayedSample)

    assert len(dataset.references()) == 2
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.probes()) == 8
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.references(group="eval")) == 2
    assert check_all_true(dataset.references(group="eval"), SampleSet)

    assert len(dataset.probes(group="eval")) == 8
    assert check_all_true(dataset.probes(group="eval"), SampleSet)

    assert len(dataset.all_samples(groups=None)) == 44
    assert check_all_true(dataset.all_samples(groups=None), DelayedSample)

    assert len(dataset.reference_ids(group="dev")) == 2
    assert len(dataset.reference_ids(group="eval")) == 2

    assert len(dataset.groups()) == 3


def test_lst_file_list_dev_sparse_filelist2():

    dataset = CSVDataset(
        legacy2_example_dir,
        "",
        csv_to_sample_loader=LSTToSampleLoader(
            data_loader=bob.io.base.load, dataset_original_directory="", extension="",
        ),
        is_sparse=True,
    )

    assert len(dataset.references()) == 3
    assert check_all_true(dataset.references(), SampleSet)

    assert len(dataset.probes()) == 9
    assert check_all_true(dataset.references(), SampleSet)


def test_csv_file_list_atnt():

    dataset = CSVDataset(atnt_protocol_path, "idiap_protocol")
    assert len(dataset.background_model_samples()) == 200
    assert len(dataset.references()) == 20
    assert len(dataset.probes()) == 100
    assert len(dataset.all_samples(groups=["train"])) == 200
    assert len(dataset.all_samples(groups=["dev"])) == 200
    assert len(dataset.all_samples(groups=None)) == 400


def data_loader(path):
    import bob.io.image

    return bob.io.base.load(path)


def test_csv_cross_validation_atnt():

    dataset = CSVDatasetCrossValidation(
        csv_file_name=atnt_protocol_path_cross_validation,
        random_state=0,
        test_size=0.8,
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=data_loader,
            dataset_original_directory=atnt_database_directory(),
            extension=".pgm",
        ),
    )
    assert len(dataset.background_model_samples()) == 80
    assert len(dataset.references("dev")) == 32
    assert len(dataset.probes("dev")) == 288
    assert len(dataset.all_samples(groups=None)) == 400


def run_experiment(dataset):
    def linearize(X):
        X = np.asarray(X)
        return np.reshape(X, (X.shape[0], -1))

    #### Testing it in a real recognition systems
    transformer = wrap(["sample"], make_pipeline(FunctionTransformer(linearize)))

    vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, Distance())

    return vanilla_biometrics_pipeline(
        dataset.background_model_samples(), dataset.references(), dataset.probes(),
    )


def test_atnt_experiment():

    dataset = CSVDataset(
        dataset_protocol_path=atnt_protocol_path,
        protocol_name="idiap_protocol",
        csv_to_sample_loader=CSVToSampleLoader(
            data_loader=data_loader,
            dataset_original_directory=atnt_database_directory(),
            extension=".pgm",
        ),
    )

    scores = run_experiment(dataset)
    assert len(scores) == 100
    assert np.alltrue([len(s) == 20] for s in scores)


def test_atnt_experiment_cross_validation():

    samples_per_identity = 10
    total_identities = 40
    samples_for_enrollment = 1

    def run_cross_validation_experiment(test_size=0.9):
        dataset = CSVDatasetCrossValidation(
            csv_file_name=atnt_protocol_path_cross_validation,
            random_state=0,
            test_size=test_size,
            csv_to_sample_loader=CSVToSampleLoader(
                data_loader=data_loader,
                dataset_original_directory=atnt_database_directory(),
                extension=".pgm",
            ),
        )

        scores = run_experiment(dataset)
        assert len(scores) == int(
            total_identities
            * test_size
            * (samples_per_identity - samples_for_enrollment)
        )

    run_cross_validation_experiment(test_size=0.9)
    run_cross_validation_experiment(test_size=0.8)
    run_cross_validation_experiment(test_size=0.5)


####
# Testing the Legacy file list
####


def test_query():
    db = FileListBioDatabase(
        legacy_example_dir, "test", use_dense_probe_file_list=False
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

    assert len(db.tclient_ids()) == 2  # 2 client ids for T-Norm score normalization
    assert len(db.zclient_ids()) == 2  # 2 client ids for Z-Norm score normalization

    assert len(db.model_ids_with_protocol()) == 6  # 6 model ids for world, dev and eval
    assert len(db.model_ids_with_protocol(groups="world")) == 2  # 2 model ids for world
    assert (
        len(db.model_ids_with_protocol(groups="optional_world_1")) == 2
    )  # 2 model ids for optional world 1
    assert (
        len(db.model_ids_with_protocol(groups="optional_world_2")) == 2
    )  # 2 model ids for optional world 2
    assert len(db.model_ids_with_protocol(groups="dev")) == 2  # 2 model ids for dev
    assert len(db.model_ids_with_protocol(groups="eval")) == 2  # 2 model ids for eval

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

    assert len(db.tobjects(groups="dev")) == 8  # 8 samples for enrolling T-norm models
    assert (
        len(db.tobjects(groups="dev", model_ids="7")) == 4
    )  # 4 samples for enrolling T-norm model '7'
    assert (
        len(db.tobjects(groups="dev", model_ids="3")) == 0
    )  # 0 samples for enrolling T-norm model '3' (no T-Norm model)
    assert len(db.zobjects(groups="dev")) == 8  # 8 samples for Z-norm impostor accesses

    assert db.client_id_from_model_id("1", group=None) == "1"
    assert db.client_id_from_model_id("3", group=None) == "3"
    assert db.client_id_from_model_id("6", group=None) == "6"
    assert db.client_id_from_t_model_id("7", group=None) == "7"


def test_query_protocol():
    db = FileListBioDatabase(
        os.path.dirname(legacy_example_dir),
        "test",
        protocol="example_filelist",
        use_dense_probe_file_list=False,
    )

    assert (
        len(db.groups()) == 5
    )  # 5 groups (dev, eval, world, optional_world_1, optional_world_2)

    assert len(db.client_ids()) == 6  # 6 client ids for world, dev and eval
    assert len(db.client_ids(groups="world",)) == 2  # 2 client ids for world
    assert (
        len(db.client_ids(groups="optional_world_1",)) == 2
    )  # 2 client ids for optional world 1
    assert (
        len(db.client_ids(groups="optional_world_2",)) == 2
    )  # 2 client ids for optional world 2
    assert len(db.client_ids(groups="dev",)) == 2  # 2 client ids for dev
    assert len(db.client_ids(groups="eval",)) == 2  # 2 client ids for eval

    assert len(db.tclient_ids()) == 2  # 2 client ids for T-Norm score normalization
    assert len(db.zclient_ids()) == 2  # 2 client ids for Z-Norm score normalization

    assert len(db.model_ids_with_protocol()) == 6  # 6 model ids for world, dev and eval
    assert (
        len(db.model_ids_with_protocol(groups="world",)) == 2
    )  # 2 model ids for world
    assert (
        len(db.model_ids_with_protocol(groups="optional_world_1",)) == 2
    )  # 2 model ids for optional world 1
    assert (
        len(db.model_ids_with_protocol(groups="optional_world_2",)) == 2
    )  # 2 model ids for optional world 2
    assert len(db.model_ids_with_protocol(groups="dev",)) == 2  # 2 model ids for dev
    assert len(db.model_ids_with_protocol(groups="eval",)) == 2  # 2 model ids for eval

    assert (
        len(db.tmodel_ids_with_protocol()) == 2
    )  # 2 model ids for T-Norm score normalization

    assert len(db.objects(groups="world",)) == 8  # 8 samples in the world set

    assert (
        len(db.objects(groups="dev", purposes="enroll",)) == 8
    )  # 8 samples for enrollment in the dev set
    assert (
        len(db.objects(groups="dev", purposes="enroll", model_ids="3",)) == 4
    )  # 4 samples for to enroll model '3' in the dev set
    assert (
        len(db.objects(groups="dev", purposes="enroll", model_ids="7",)) == 0
    )  # 0 samples for enrolling model '7' (it is a T-Norm model)
    assert (
        len(db.objects(groups="dev", purposes="probe",)) == 8
    )  # 8 samples as probes in the dev set
    assert (
        len(db.objects(groups="dev", purposes="probe", classes="client",)) == 8
    )  # 8 samples as client probes in the dev set
    assert (
        len(db.objects(groups="dev", purposes="probe", classes="impostor",)) == 4
    )  # 4 samples as impostor probes in the dev set

    assert len(db.tobjects(groups="dev",)) == 8  # 8 samples for enrolling T-norm models
    assert (
        len(db.tobjects(groups="dev", model_ids="7",)) == 4
    )  # 4 samples for enrolling T-norm model '7'
    assert (
        len(db.tobjects(groups="dev", model_ids="3",)) == 0
    )  # 0 samples for enrolling T-norm model '3' (no T-Norm model)
    assert len(db.zobjects(groups="dev")) == 8  # 8 samples for Z-norm impostor accesses

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
    nose.tools.assert_raises(
        ValueError, db.model_ids_with_protocol, protocol=prot, groups="eval"
    )  # eval does not exist for this protocol
    assert len(db.objects(protocol=prot, groups="dev", purposes="enroll")) == 12
    assert len(db.objects(protocol=prot, groups="dev", purposes="probe")) == 9


def test_noztnorm():
    db = FileListBioDatabase(
        os.path.join(os.path.dirname(legacy_example_dir), "example_filelist2"), "test"
    )
    assert len(db.all_files())


def test_query_dense():
    db = FileListBioDatabase(legacy_example_dir, "test", use_dense_probe_file_list=True)

    assert len(db.objects(groups="world")) == 8  # 8 samples in the world set

    assert (
        len(db.objects(groups="dev", purposes="enroll")) == 8
    )  # 8 samples for enrollment in the dev set
    assert (
        len(db.objects(groups="dev", purposes="probe")) == 8
    )  # 8 samples as probes in the dev set


def test_annotation():
    db = FileListBioDatabase(
        legacy_example_dir,
        "test",
        use_dense_probe_file_list=False,
        annotation_directory=legacy_example_dir,
        annotation_type="named",
    )
    f = [o for o in db.objects() if o.path == "data/model4_session1_sample2"][0]
    annots = db.annotations(f)

    assert annots is not None
    assert "key1" in annots
    assert "key2" in annots
    assert annots["key1"] == (20, 10)
    assert annots["key2"] == (40, 30)


def test_multiple_extensions():
    # check that the old behavior still works
    db = FileListBioDatabase(
        legacy_example_dir,
        "test",
        use_dense_probe_file_list=False,
        original_directory=legacy_example_dir,
        original_extension=".pos",
    )
    file = bob.bio.base.database.BioFile(
        4, "data/model4_session1_sample2", "data/model4_session1_sample2"
    )
    file_name = db.original_file_name(file, True)
    assert file_name == os.path.join(legacy_example_dir, file.path + ".pos")

    # check that the new behavior works as well
    db = FileListBioDatabase(
        legacy_example_dir,
        "test",
        use_dense_probe_file_list=False,
        original_directory=legacy_example_dir,
        original_extension=[".jpg", ".pos"],
    )
    file_name = db.original_file_name(file)
    assert file_name == os.path.join(legacy_example_dir, file.path + ".pos")

    file = bob.bio.base.database.BioFile(
        4, "data/model4_session1_sample1", "data/model4_session1_sample1"
    )
    nose.tools.assert_raises(IOError, db.original_file_name, file, False)


def test_driver_api():
    from bob.db.base.script.dbmanage import main

    assert (
        main(
            (
                "bio_filelist dumplist --list-directory=%s --self-test"
                % legacy_example_dir
            ).split()
        )
        == 0
    )
    assert (
        main(
            (
                "bio_filelist dumplist --list-directory=%s --purpose=enroll --group=dev --class=client --self-test"
                % legacy_example_dir
            ).split()
        )
        == 0
    )
    assert (
        main(
            (
                "bio_filelist checkfiles --list-directory=%s --self-test"
                % legacy_example_dir
            ).split()
        )
        == 0
    )

