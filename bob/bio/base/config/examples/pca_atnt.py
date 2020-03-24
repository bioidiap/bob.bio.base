#from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, AlgorithmAdaptor

import bob.bio.face
from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector
database = DatabaseConnector(bob.bio.face.database.AtntBioDatabase(original_directory="./atnt"), protocol="Default")

from sklearn.pipeline import Pipeline, make_pipeline
from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from bob.bio.base.transformers import CheckpointSampleLinearize, CheckpointSamplePCA


from bob.pipelines.mixins import dask_it
extractor = Pipeline(steps=[('0',CheckpointSampleLinearize(features_dir="./example/extractor0")), 
	                        ('1',CheckpointSamplePCA(features_dir="./example/extractor1", model_path="./example/pca.pkl"))])
#extractor = dask_it(extractor)

from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import Distance, BiometricAlgorithmCheckpointMixin
class CheckpointDistance(BiometricAlgorithmCheckpointMixin, Distance):  pass
algorithm = CheckpointDistance(features_dir="./example/")
#algorithm = Distance()
