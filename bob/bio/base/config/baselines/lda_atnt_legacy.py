# from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, AlgorithmAdaptor

import bob.db.atnt
from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector

database = DatabaseConnector(bob.db.atnt.Database(), protocol="Default")

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA

from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from bob.bio.base.mixins import CheckpointSampleLinearize
from bob.bio.base.mixins.legacy import LegacyProcessorMixin, LegacyAlgorithmMixin


class CheckpointSamplePCA(CheckpointMixin, SampleMixin, PCA):
    """
    Enables SAMPLE and CHECKPOINTIN handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    pass


#### PREPROCESSOR LEGACY ###
import functools

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
import bob.bio.face

face_cropper = functools.partial(
    bob.bio.face.preprocessor.FaceCrop,
    cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
    fixed_positions={"leye": FIXED_LEFT_EYE_POS, "reye": FIXED_RIGHT_EYE_POS},
)

from bob.pipelines.mixins import mix_me_up
preprocessor = mix_me_up((CheckpointMixin, SampleMixin), LegacyProcessorMixin)

#### ALGORITHM LEGACY #####

algorithm = functools.partial(bob.bio.base.algorithm.LDA, use_pinv=True, pca_subspace_dimension=0.90)

from bob.pipelines.mixins import dask_it

extractor = Pipeline(
    steps=[
        ("0", preprocessor(callable=face_cropper, features_dir="./example/extractor0")),
        ("1", CheckpointSampleLinearize(features_dir="./example/extractor1")),
        (
            "2",
            LegacyAlgorithmMixin(
                callable=algorithm, features_dir="./example/extractor2", model_path="./example/"
            ),
        ),
    ]
)

extractor = dask_it(extractor)

from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import (
    Distance,
    BiometricAlgorithmCheckpointMixin,
)


class CheckpointDistance(BiometricAlgorithmCheckpointMixin, Distance):
    pass


algorithm = CheckpointDistance(features_dir="./example/")
# algorithm = Distance()
