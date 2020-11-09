import os
from bob.bio.base.test.dummy.database import database

import bob.io.image

# Creates a list of unique str
samples = [s.key for s in database.background_model_samples()]

def reader(sample):
    data = bob.io.image.load(
        os.path.join(database.database.original_directory, sample + database.database.original_extension)
    )
    return data


def make_key(sample):
    return sample