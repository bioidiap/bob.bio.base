from bob.bio.base.pipelines.blocks import DatabaseConnector, AlgorithmAdaptor
import functools
import bob.db.atnt

database = DatabaseConnector(bob.db.atnt.Database(), protocol="Default")

from bob.bio.face.preprocessor import Base
preprocessor = functools.partial(
                Base,
                color_channel="gray",
                dtype="float64",
            )


from bob.bio.base.extractor import Linearize
extractor = Linearize
#extractor = 'linearize'


from bob.bio.base.algorithm import PCA
algorithm = AlgorithmAdaptor(functools.partial(PCA, 0.99))
