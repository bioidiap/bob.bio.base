from bob.bio.face.database import AtntBioDatabase
from bob.bio.base.algorithm import LDA
from bob.bio.face.preprocessor import FaceCrop
from sklearn.pipeline import make_pipeline
from bob.pipelines.transformers import CheckpointSampleLinearize
from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, Preprocessor, AlgorithmAsTransformer, AlgorithmAsBioAlg
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

base_dir = "example"

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
    Preprocessor(callable=face_cropper, features_dir=os.path.join(base_dir,"transformer0")),
    CheckpointSampleLinearize(features_dir=os.path.join(base_dir,"transformer1")),
    AlgorithmAsTransformer(
        callable=lda, features_dir=os.path.join(base_dir,"transformer2"), model_path=os.path.join(base_dir, "lda.hdf5")
    ),
)


algorithm = AlgorithmAsBioAlg(callable=lda, features_dir="./example/")


from bob.pipelines.mixins import estimator_dask_it, mix_me_up
from bob.bio.base.pipelines.vanilla_biometrics.mixins import (
    BioAlgDaskMixin,
)

transformer = estimator_dask_it(transformer)
algorithm = mix_me_up([BioAlgDaskMixin], algorithm)
