from bob.bio.base.pipelines.vanilla_biometrics.legacy import DatabaseConnector, AlgorithmAdaptor

import bob.db.atnt

database = DatabaseConnector(bob.db.atnt.Database(), protocol="Default")

preprocessor = "face-detect"

extractor = 'linearize'

algorithm = 'pca'
