import os

import bob.io.base

from bob.bio.base.test.dummy.database import database

# Creates a list of unique str
samples = [s.key for s in database.background_model_samples()]


def reader(sample):
    data = bob.io.base.load(
        os.path.join(
            database.original_directory, sample + database.original_extension
        )
    )
    return data


def make_key(sample):
    return sample
