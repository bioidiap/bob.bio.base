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
# A generic implementation is available in:
# from bob.bio.base.algorithm import Distance
class EuclideanDistance(BioAlgorithm):
    def create_templates(self, list_of_feature_sets, enroll):
        # list_of_feature_sets is a list of lists of features with the shape of Nx?xD.
        # The second dimension can be variable in list_of_feature_sets depening on the
        # datbase protocol. Hence, you cannot call:
        # list_of_feature_sets = np.array(list_of_feature_sets)
        templates = []
        for feature_set in list_of_feature_sets:
            # template is a D dimensional vector
            template = np.mean(feature_set, axis=0)
            templates.append(template)
        # templates is an NxD dimensional array
        return templates

    def compare(self, enroll_templates, probe_templates):
        # enroll_templates is an NxD dimensional array
        # probe_templates is an MxD dimensional array
        # scores will be an NxM dimensional array
        scores = []
        for model in enroll_templates:
            scores.append([])
            for probe in probe_templates:
                similarity = 1 / np.linalg.norm(model - probe)
                scores[-1].append(similarity)
        scores = np.array(scores, dtype=float)
        return scores


bio_algorithm = EuclideanDistance()

# Creation of the pipeline #
# `pipeline` will be used by the `bob bio pipeline simple` command
pipeline = PipelineSimple(transformer, bio_algorithm)

# you can also specify the other options in this file:
database = "atnt"
output = "results"
