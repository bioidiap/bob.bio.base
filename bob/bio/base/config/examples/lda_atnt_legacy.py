from bob.bio.face.database import AtntBioDatabase
from bob.bio.base.algorithm import LDA
from bob.bio.face.preprocessor import FaceCrop
from sklearn.pipeline import make_pipeline
from bob.bio.base.mixins.legacy import (
    LegacyPreprocessor,
    LegacyAlgorithmAsTransformer,
)
from bob.pipelines.transformers import CheckpointSampleLinearize
from bob.bio.base.pipelines.vanilla_biometrics.legacy import LegacyDatabaseConnector
import functools
from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import (
    CheckpointDistance,
)

# DATABASE

database = LegacyDatabaseConnector(
    AtntBioDatabase(original_directory="./atnt", protocol="Default"),
)


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

# ALGORITHM LEGACY

lda = functools.partial(LDA, use_pinv=True, pca_subspace_dimension=0.90)


transformer = make_pipeline(
    LegacyPreprocessor(callable=face_cropper, features_dir="./example/transformer0"),
    CheckpointSampleLinearize(features_dir="./example/transformer1"),
    LegacyAlgorithmAsTransformer(
        callable=lda, features_dir="./example/transformer2", model_path="./example/"
    ),
)


algorithm = CheckpointDistance(features_dir="./example/")


# comment out the code below to disable dask
from bob.pipelines.mixins import estimator_dask_it, mix_me_up
from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import (
    BioAlgDaskMixin,
)

transformer = estimator_dask_it(transformer)
algorithm = mix_me_up([BioAlgDaskMixin], algorithm)
