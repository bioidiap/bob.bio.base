import numpy

## Implementation of a Transformer

from sklearn.base import TransformerMixin, BaseEstimator

class CustomTransformer(TransformerMixin, BaseEstimator):
    def transform(self, X):
        transformed_X = X
        return transformed_X

    def fit(self, X, y=None):
        return self


## Implementation of the BioAlgorithm

from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import BioAlgorithm

class CustomDistance(BioAlgorithm):
    def enroll(self, enroll_features):
        model = numpy.mean(enroll_features, axis=0)
        return model

    def score(self, model, probe):
        distance = 1/numpy.linalg.norm(model-probe)
        return distance


## Creation of the pipeline

from sklearn.pipeline import make_pipeline
from bob.pipelines import wrap
from bob.bio.base.pipelines.vanilla_biometrics import VanillaBiometricsPipeline

# Instantiate the Transformers
my_transformer = CustomTransformer()

# Chain the Transformers together
transformer = make_pipeline(
    wrap(["sample"], my_transformer),
    # Add more transformers here if needed
)

# Instantiate the BioAlgorithm
bio_algorithm = CustomDistance()

# Assemble the Vanilla Biometric pipeline and execute
pipeline = VanillaBiometricsPipeline(transformer, bio_algorithm)

# Prevent the need to implement a `score_multiple_biometric_references` method
database.allow_scoring_with_all_biometric_references = False


# `pipeline` will be used by the `bob bio pipelines` command
