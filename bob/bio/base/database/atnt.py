#!/usr/bin/env python
"""
ATNT database implementation
"""

from pathlib import Path

from sklearn.pipeline import make_pipeline

import bob.io.base
import bob.io.base.test_utils

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.extension.download import get_file


class AtntBioDatabase(CSVDataset):
    """
    The AT&T (aka ORL) database of faces
    (http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html). This
    class defines a simple protocol for training, enrollment and probe by
    splitting the few images of the database in a reasonable manner. Due to the
    small size of the database, there is only a 'dev' group, and I did not
    define an 'eval' group.
    """

    def __init__(
        self,
        protocol="idiap_protocol",
        dataset_original_directory=None,
        **kwargs,
    ):

        # Downloading model if not exists
        dataset_protocol_path = bob.io.base.test_utils.datafile(
            "atnt", "bob.bio.base.test", "data"
        )
        if dataset_original_directory is None:
            path = get_file(
                "atnt_faces.zip",
                ["http://www.idiap.ch/software/bob/data/bob/att_faces.zip"],
                file_hash="6efb25cb0d40755e9492b9c012e3348d",
                cache_subdir="datasets/atnt",
                extract=True,
            )
            dataset_original_directory = str(Path(path).parent)

        super().__init__(
            name="atnt",
            dataset_protocol_path=dataset_protocol_path,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=dataset_original_directory,
                    extension=".pgm",
                ),
            ),
            **kwargs,
        )
        # just expost original_directory for backward compatibility of tests
        self.original_directory = dataset_original_directory
        self.original_extension = ".pgm"

    # define an objects method for compatibility with the old tests
    def objects(
        self, model_ids=None, groups=None, purposes=None, protocol=None
    ):
        samples = []

        if groups is None:
            groups = self.groups()

        if "train" in groups or "world" in groups:
            samples += self.background_model_samples()

        if purposes is None:
            purposes = ("enroll", "probe")

        if "enroll" in purposes:
            samples += self.references()

        if "probe" in purposes:
            samples += self.probes()

        if model_ids:
            samples = [s for s in samples if s.reference_id in model_ids]

        # create the old attributes
        for s in samples:
            s.client_id = s.reference_id
            s.path = s.id = s.key

        return samples


def main():
    """Code used to generate the .csv files for atnt database"""
    import os

    from csv import DictWriter

    from bob.bio.face.database.atnt import AtntBioDatabase

    database = AtntBioDatabase()

    all_protocols = ["Default"]

    for protocol in all_protocols:
        # Retrieve the file lists from the legacy db
        train_files = database.objects(
            groups=["world"], protocol=protocol, purposes=["enroll"]
        )
        dev_enroll = database.objects(
            groups=["dev"], protocol=protocol, purposes=["enroll"]
        )
        dev_probe = database.objects(
            groups=["dev"], protocol=protocol, purposes=["probe"]
        )
        eval_enroll = []
        eval_probe = []

        # Check that the lists are not empty
        has_eval, has_train = True, True
        if not all([eval_enroll, eval_probe]):
            has_eval = False
        if not train_files:
            has_train = False

        # Create the folder structure
        protocol_path = os.path.join("atnt", protocol)
        dev_path = os.path.join(protocol_path, "dev")
        os.makedirs(dev_path, exist_ok=True)
        if has_eval:
            eval_path = os.path.join(protocol_path, "eval")
            os.makedirs(eval_path, exist_ok=True)
        if has_train:
            train_path = os.path.join(protocol_path, "norm")
            os.makedirs(train_path, exist_ok=True)

        # Writing the CSV files
        def write_to_csv(path, filelist, header, fields):
            with open(path, "w") as f:
                csv_writer = DictWriter(f, delimiter=",", fieldnames=header)
                csv_writer.writeheader()
                csv_writer.writerows(
                    [
                        {
                            k: v
                            for k, v in zip(
                                header, [getattr(s, a, None) for a in fields]
                            )
                        }
                        for s in filelist
                    ]
                )

        # Columns in the csv (header)
        csv_fields = ["PATH", "REFERENCE_ID", "ID", "ANNOTATIONS"]
        # Corresponding fields in the File objects
        file_attr = ["path", "client_id", "id", "no_annotations"]

        # Probe header have some special metadata/columns
        csv_fields_probes = csv_fields
        file_attr_probes = file_attr

        write_to_csv(
            os.path.join(dev_path, "for_models.csv"),
            dev_enroll,
            csv_fields,
            file_attr,
        )
        write_to_csv(
            os.path.join(dev_path, "for_probes.csv"),
            dev_probe,
            csv_fields_probes,
            file_attr_probes,
        )
        if has_eval:
            write_to_csv(
                os.path.join(eval_path, "for_models.csv"),
                eval_enroll,
                csv_fields,
                file_attr,
            )
            write_to_csv(
                os.path.join(eval_path, "for_probes.csv"),
                eval_probe,
                csv_fields_probes,
                file_attr_probes,
            )
        if has_train:
            write_to_csv(
                os.path.join(train_path, "train_world.csv"),
                train_files,
                csv_fields,
                file_attr,
            )

    print("Created 'atnt' folder.")


if __name__ == "__main__":
    main()
