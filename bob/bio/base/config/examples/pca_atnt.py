from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector
from sklearn.pipeline import make_pipeline
from bob.pipelines.transformers import CheckpointSampleLinearize, CheckpointSamplePCA
from bob.bio.base.pipelines.vanilla_biometrics.implemented import (
    CheckpointDistance,
)
from bob.bio.face.database import AtntBioDatabase
import os


base_dir = "example"

database = DatabaseConnector(AtntBioDatabase(original_directory="./atnt", protocol="Default"))

transformer = make_pipeline(
    CheckpointSampleLinearize(features_dir=os.path.join(base_dir, "linearize")),
    CheckpointSamplePCA(
        features_dir=os.path.join(base_dir, "pca_features"), model_path=os.path.join(base_dir, "pca.pkl")
    ),
)
algorithm = CheckpointDistance(features_dir=base_dir)

# # comment out the code below to disable dask
from bob.pipelines.mixins import estimator_dask_it, mix_me_up
from bob.bio.base.pipelines.vanilla_biometrics.mixins import (
    BioAlgDaskMixin,
)

transformer = estimator_dask_it(transformer)
algorithm = mix_me_up([BioAlgDaskMixin], algorithm)

