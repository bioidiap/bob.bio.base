from sklearn.base import BaseEstimator

from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import PipelineSimple
from bob.pipelines import wrap


class DummyTransformer(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [X[0].mean(axis=1)]


transformer = wrap(["sample"], DummyTransformer())
biometric_algorithm = Distance()

pipeline = PipelineSimple(transformer, biometric_algorithm)
