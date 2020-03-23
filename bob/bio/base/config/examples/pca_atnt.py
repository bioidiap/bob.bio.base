#from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, AlgorithmAdaptor

import bob.db.atnt
from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector
database = DatabaseConnector(bob.db.atnt.Database(), protocol="Default")

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA

from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from bob.bio.base.mixins import CheckpointSampleLinearize

class CheckpointSamplePCA(CheckpointMixin, SampleMixin, PCA):
    """
    Enables SAMPLE and CHECKPOINTIN handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    pass

#extractor = make_pipeline([CheckpointSampleLinearize(), CheckpointSamplePCA()])
from bob.pipelines.mixins import dask_it
extractor = Pipeline(steps=[('0',CheckpointSampleLinearize(features_dir="./example/extractor0")), 
	                        ('1',CheckpointSamplePCA(features_dir="./example/extractor1", model_path="./example/pca.pkl"))])
#extractor = dask_it(extractor)

from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import Distance, BiometricAlgorithmCheckpointMixin
class CheckpointDistance(BiometricAlgorithmCheckpointMixin, Distance):  pass
algorithm = CheckpointDistance(features_dir="./example/")
#algorithm = Distance()
