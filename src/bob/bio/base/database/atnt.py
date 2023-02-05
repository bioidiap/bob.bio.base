#!/usr/bin/env python
"""
ATNT database implementation
"""

from pathlib import Path

from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database.utils import download_file, md5_hash

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

    category = "base"
    dataset_protocols_urls = [
        "https://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz",
        "http://www.idiap.ch/software/bob/databases/latest/base/atnt-f529acef.tar.gz",
    ]
    dataset_protocols_checksum = "f529acef"

    def __init__(
        self,
        protocol="idiap_protocol",
        dataset_original_directory=None,
        **kwargs,
    ):
        """Custom init to download the raw data files."""

        # Download the raw data (or use a cached local version)
        if dataset_original_directory is None:
            path = download_file(
                urls=[
                    "http://www.idiap.ch/software/bob/data/bob/att_faces.zip"
                ],
                destination_sub_directory="datasets/atnt",
                destination_filename="atnt_faces.zip",
                checksum="6efb25cb0d40755e9492b9c012e3348d",
                checksum_fct=md5_hash,
                extract=True,
            )
            dataset_original_directory = Path(path).as_posix()

        super().__init__(
            name="atnt",
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
