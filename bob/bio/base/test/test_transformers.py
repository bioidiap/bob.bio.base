#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# @author: Tiago de Freitas Pereira <tiago.pereira@idiap.ch>

from bob.pipelines.sample import Sample, SampleSet, DelayedSample
import os
import numpy
import tempfile
from sklearn.utils.validation import check_is_fitted

from bob.bio.base.transformers import Linearize, SampleLinearize, CheckpointSampleLinearize
def test_linearize_processor():
    
    ## Test the transformer only
    transformer = Linearize()
    X = numpy.zeros(shape=(10,10))
    X_tr = transformer.transform(X)
    assert X_tr.shape == (100,)


    ## Test wrapped in to a Sample
    sample = Sample(X, key="1")
    transformer = SampleLinearize()
    X_tr = transformer.transform([sample])
    assert X_tr[0].data.shape == (100,)

    ## Test checkpoint    
    with tempfile.TemporaryDirectory() as d:
        transformer = CheckpointSampleLinearize(features_dir=d)
        X_tr =  transformer.transform([sample])
        assert X_tr[0].data.shape == (100,)
        assert os.path.exists(os.path.join(d, "1.h5"))


from bob.bio.base.transformers import SamplePCA, CheckpointSamplePCA
def test_pca_processor():
    
    ## Test wrapped in to a Sample
    X = numpy.random.rand(100,10)
    samples = [Sample(data, key=str(i)) for i, data in enumerate(X)]

    # fit
    n_components = 2
    estimator = SamplePCA(n_components=n_components)
    estimator = estimator.fit(samples)
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
    assert check_is_fitted(estimator, "n_components_") is None
    
    # transform
    samples_tr = estimator.transform(samples)
    assert samples_tr[0].data.shape == (n_components,)
    

    ## Test Checkpoining
    with tempfile.TemporaryDirectory() as d:        
        model_path = os.path.join(d, "model.pkl")
        estimator = CheckpointSamplePCA(n_components=n_components, features_dir=d, model_path=model_path)

        # fit
        estimator = estimator.fit(samples)
        assert check_is_fitted(estimator, "n_components_") is None
        assert os.path.exists(model_path)
        
        # transform
        samples_tr = estimator.transform(samples)
        assert samples_tr[0].data.shape == (n_components,)        
        assert os.path.exists(os.path.join(d, samples_tr[0].key+".h5"))
