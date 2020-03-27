from bob.bio.base.pipelines.vanilla_biometrics.implemented import CheckpointDistance
from bob.bio.base.pipelines.vanilla_biometrics.legacy import (
    DatabaseConnector,
    Preprocessor,
    Extractor,
    AlgorithmAsBioAlg,
)
from bob.bio.face.database.mobio import MobioBioDatabase
from bob.bio.face.preprocessor import FaceCrop
from bob.extension import rc
from bob.pipelines.transformers import CheckpointSampleLinearize, CheckpointSamplePCA
from sklearn.pipeline import make_pipeline
import functools
import os
import bob.bio.face
import math

base_dir = "example"


database = DatabaseConnector(
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
    bob.bio.face.preprocessor.INormLBP,
    face_cropper=functools.partial(
        bob.bio.face.preprocessor.FaceCrop,
        cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
        cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
        color_channel="gray",
    ),
)

extractor = functools.partial(
    bob.bio.face.extractor.GridGraph,
    # Gabor parameters
    gabor_sigma=math.sqrt(2.0) * math.pi,
    # what kind of information to extract
    normalize_gabor_jets=True,
    # setup of the fixed grid
    node_distance=(8, 8),
)


transformer = make_pipeline(
    Preprocessor(preprocessor, features_dir=os.path.join(base_dir, "face_cropper")),
    Extractor(extractor, features_dir=os.path.join(base_dir, "gabor_graph")),
)


## algorithm

gabor_jet = functools.partial(
    bob.bio.face.algorithm.GaborJet,
    gabor_jet_similarity_type="PhaseDiffPlusCanberra",
    multiple_feature_scoring="max_jet",
    gabor_sigma=math.sqrt(2.0) * math.pi,
)

algorithm = AlgorithmAsBioAlg(callable=gabor_jet, features_dir=base_dir)


# comment out the code below to disable dask
from bob.pipelines.mixins import estimator_dask_it, mix_me_up
from bob.bio.base.pipelines.vanilla_biometrics.mixins import BioAlgDaskMixin

# transformer = estimator_dask_it(transformer)
# algorithm = mix_me_up([BioAlgDaskMixin], algorithm)
