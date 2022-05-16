import logging
import pickle

from bob.bio.base.pipelines import BioAlgorithm
from bob.learn.em import JFAMachine

logger = logging.getLogger(__name__)


class JFA(JFAMachine, BioAlgorithm):
    """JFA transformer and bioalgorithm to be used in pipelines"""

    def transform(self, X):
        """Passthrough"""
        return X

    @classmethod
    def custom_enrolled_save_fn(cls, data, path):
        pickle.dump(data, open(path, "wb"))

    @classmethod
    def custom_enrolled_load_fn(cls, path):
        return pickle.load(open(path, "rb"))

    def _more_tags(self):
        return {
            "bob_fit_supports_dask_bag": True,
            "bob_fit_extra_input": [("y", "reference_id_int")],
            "bob_enrolled_save_fn": self.custom_enrolled_save_fn,
            "bob_enrolled_load_fn": self.custom_enrolled_load_fn,
            "bob_checkpoint_features": False,
        }
