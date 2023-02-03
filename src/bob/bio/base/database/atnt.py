#!/usr/bin/env python
"""
ATNT database implementation
"""

from pathlib import Path

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.extension.download import get_file

from . import CSVDatabase, FileSampleLoader


class AtntBioDatabase(CSVDatabase):
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
        # Download the protocol definition file
        dataset_protocols_path = get_file(
            "atnt_protocols.tar.gz",
            [
                "https://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz"
            ],
            file_hash="f529acef",
        )
        # Download the raw data (or use a cached local version)
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
            dataset_protocols_path=dataset_protocols_path,
            protocol=protocol,
            transformer=make_pipeline(
                FileSampleLoader(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=dataset_original_directory,
                    extension=".pgm",
                ),
            ),
            **kwargs,
        )
        # just expose original_directory for backward compatibility of tests
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
            samples = [s for s in samples if s.template_id in model_ids]

        # create the old attributes
        for s in samples:
            s.client_id = s.template_id
            s.path = s.id = s.key

        return samples
