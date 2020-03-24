
import functools
import bob.db.atnt
from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, DatabaseConnectorAnnotated
from bob.extension import rc
import bob.bio.face

from bob.bio.base.mixins.legacy import LegacyProcessorMixin, LegacyAlgorithmMixin
from bob.bio.base.pipelines.vanilla_biometrics.legacy import LegacyBiometricAlgorithm

import os
base_dir = "/idiap/temp/tpereira/mobio/pca"
#base_dir = "./example"


original_directory=rc['bob.db.mobio.directory']
annotation_directory=rc['bob.db.mobio.annotation_directory']
database = DatabaseConnectorAnnotated(bob.bio.face.database.mobio.MobioBioDatabase(
	                         original_directory=original_directory,
	                         annotation_directory=annotation_directory,
	                         original_extension=".png"
	                         ), 
	                         protocol="mobile0-male")

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA

from bob.pipelines.mixins import CheckpointMixin, SampleMixin
from bob.bio.base.mixins import CheckpointSampleLinearize

class CheckpointSamplePCA(CheckpointMixin, SampleMixin, PCA):
    """
    Enables SAMPLE and CHECKPOINTIN handling for https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """
    pass



#### PREPROCESSOR LEGACY ###

# Using face crop
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH = CROPPED_IMAGE_HEIGHT * 4 // 5

## eye positions for frontal images
RIGHT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 - 1)
LEFT_EYE_POS = (CROPPED_IMAGE_HEIGHT // 5, CROPPED_IMAGE_WIDTH // 4 * 3)

original_preprocessor = functools.partial(
                  bob.bio.face.preprocessor.FaceCrop,
                  cropped_image_size=(CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
                  cropped_positions={"leye": LEFT_EYE_POS, "reye": RIGHT_EYE_POS},
               )


from bob.pipelines.mixins import mix_me_up
preprocessor = mix_me_up((CheckpointMixin, SampleMixin), LegacyProcessorMixin)

from bob.pipelines.mixins import dask_it
extractor = Pipeline(steps=[
	                        ('0', preprocessor(callable=original_preprocessor, features_dir=os.path.join(base_dir,"extractor0"))),
                            ('1',CheckpointSampleLinearize(features_dir=os.path.join(base_dir,"extractor1"))), 
	                        ('2',CheckpointSamplePCA(features_dir=os.path.join(base_dir,"extractor2"), model_path=os.path.join(base_dir,"pca.pkl")))
	                       ])
extractor = dask_it(extractor)

from bob.bio.base.pipelines.vanilla_biometrics.biometric_algorithm import Distance, BiometricAlgorithmCheckpointMixin

class CheckpointDistance(BiometricAlgorithmCheckpointMixin, Distance):  pass
algorithm = CheckpointDistance(features_dir=base_dir)
#algorithm = Distance()
