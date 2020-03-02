from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, AlgorithmAdaptor

import bob.db.atnt

database = DatabaseConnector(bob.db.atnt.Database(), protocol="Default")

preprocessor = "face-detect"

extractor = 'linearize'


from bob.bio.base.algorithm import PCA
import functools
algorithm = AlgorithmAdaptor(functools.partial(PCA,0.99))
