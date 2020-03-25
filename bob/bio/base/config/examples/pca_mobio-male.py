from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import (
    CheckpointDistance,
)
from bob.bio.base.pipelines.vanilla_biometrics.legacy import (
    LegacyDatabaseConnector,
    LegacyPreprocessor,
)
from bob.bio.face.database.mobio import MobioBioDatabase
from bob.bio.face.preprocessor import FaceCrop
from bob.extension import rc
from bob.pipelines.transformers import CheckpointSampleLinearize, CheckpointSamplePCA
from sklearn.pipeline import make_pipeline
import functools


database = LegacyDatabaseConnector(
    MobioBioDatabase(
        original_directory=rc["bob.db.mobio.directory"],
        annotation_directory=rc["bob.db.mobio.annotation_directory"],
        original_extension=".png",
        protocol="mobile0-male",
    )
)

# Using face crop
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5
# eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)
# FaceCrop
preprocessor = functools.partial(
    FaceCrop,
    cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
)

transformer = make_pipeline(
    LegacyPreprocessor(preprocessor),
    CheckpointSampleLinearize(features_dir="./example/extractor0"),
    CheckpointSamplePCA(
        features_dir="./example/extractor1", model_path="./example/pca.pkl"
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
