from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector
from sklearn.pipeline import make_pipeline
from bob.pipelines.transformers import CheckpointSampleLinearize, CheckpointSamplePCA
from bob.bio.base.pipelines.vanilla_biometrics.implemented import (
    CheckpointDistance,
)
from bob.bio.face.database import AtntBioDatabase


database = DatabaseConnector(
    AtntBioDatabase(original_directory="./atnt"), protocol="Default"
)
transformer = make_pipeline(
    CheckpointSampleLinearize(features_dir="./example/extractor0"),
    CheckpointSamplePCA(
        features_dir="./example/extractor1", model_path="./example/pca.pkl"
    ),
)
algorithm = CheckpointDistance(features_dir="./example/")

# comment out the code below to disable dask
from bob.pipelines.mixins import estimator_dask_it, mix_me_up
from bob.bio.base.pipelines.vanilla_biometrics.mixins import (
    BioAlgDaskMixin,
)

transformer = estimator_dask_it(transformer)
algorithm = mix_me_up(BioAlgDaskMixin, algorithm)
