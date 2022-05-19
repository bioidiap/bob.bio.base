import numpy as np

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_array

from bob.bio.base.pipelines import BioAlgorithm, PipelineSimple
from bob.pipelines import wrap

# Transformers #
pca = PCA(n_components=0.95)

# the images are in shape of Nx112x92, we want to flatten to Nx10304 them so we can train a PCA on them.
# A similar implementation is available in:
# from bob.pipelines.transformers import Linearize
def flatten(images):
    images = check_array(images, allow_nd=True)
    new_shape = [images.shape[0], -1]
    return np.reshape(images, new_shape)


flatten_transformer = FunctionTransformer(flatten, validate=False)

# Chain the Transformers together
transformer = make_pipeline(flatten_transformer, pca)

# All transformers must be sample transformers
transformer = wrap(["sample"], transformer)

# Implementation of the BioAlgorithm #
# A better implementation is available in:
# from bob.bio.base.algorithm import Distance
class EuclideanDistance(BioAlgorithm):
    def enroll(self, enroll_features):
        model = np.mean(enroll_features, axis=0)
        return model

    def score(self, model, probe):
        similarity = 1 / np.linalg.norm(model - probe)
        # you should always return a similarity score
        return similarity


bio_algorithm = EuclideanDistance()


# Creation of the pipeline #
# `pipeline` will be used by the `bob bio pipeline simple` command
pipeline = PipelineSimple(transformer, bio_algorithm)

# you can also specify the other options in this file:
database = "atnt"
output = "results"
