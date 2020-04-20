from bob.bio.face.database import AtntBioDatabase
from bob.bio.gmm.algorithm import ISV
from bob.bio.face.preprocessor import FaceCrop
from sklearn.pipeline import make_pipeline
from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, Preprocessor, AlgorithmAsTransformer, AlgorithmAsBioAlg, Extractor
import functools
from bob.bio.base.pipelines.vanilla_biometrics.implemented import (
    Distance,
    CheckpointDistance,
)
import os

# DATABASE
database = DatabaseConnector(
    AtntBioDatabase(original_directory="./atnt", protocol="Default"),
)
database.allow_scoring_with_all_biometric_references = True

base_dir = "example/isv"

# PREPROCESSOR LEGACY

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)


# RANDOM EYES POSITIONS
# I JUST MADE UP THESE NUMBERS
FIXED_RIGHT_EYE_POS = (30, 30)
FIXED_LEFT_EYE_POS = (20, 50)

face_cropper = functools.partial(
    FaceCrop,
    cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
    fixed_positions={"leye": FIXED_LEFT_EYE_POS, "reye": FIXED_RIGHT_EYE_POS},
)


import bob.bio.face

extractor = functools.partial(
    bob.bio.face.extractor.DCTBlocks,
    block_size=12,
    block_overlap=11,
    number_of_dct_coefficients=45,
)



# ALGORITHM LEGACY
isv = functools.partial(ISV, subspace_dimension_of_u=10, number_of_gaussians=2)

model_path=os.path.join(base_dir, "ubm_u.hdf5")
transformer = make_pipeline(
    Preprocessor(callable=face_cropper, features_dir=os.path.join(base_dir,"face_crop")),
    Extractor(extractor, features_dir=os.path.join(base_dir, "dcts")),
    AlgorithmAsTransformer(
        callable=isv, features_dir=os.path.join(base_dir,"isv"), model_path=model_path
    ),
)


algorithm = AlgorithmAsBioAlg(callable=isv, features_dir=base_dir, model_path=model_path)


from bob.bio.base.pipelines.vanilla_biometrics import VanillaBiometrics, dask_vanilla_biometrics

#pipeline = VanillaBiometrics(transformer, algorithm)
pipeline = dask_vanilla_biometrics(VanillaBiometrics(transformer, algorithm))
