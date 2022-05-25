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

    def create_templates(self, list_of_feature_sets, enroll):
        if enroll:
            return [
                self.enroll(feature_set) for feature_set in list_of_feature_sets
            ]
        else:
            # TODO: We should compute these parts of self.score:
            # x = self.estimate_x(data)
            # Ux = self._U @ x
            # here to make scoring faster
            return list_of_feature_sets

    def compare(self, enroll_templates, probe_templates):
        # TODO: The underlying score method actually supports batched scoring
        return [
            [self.score(enroll, probe) for probe in probe_templates]
            for enroll in enroll_templates
        ]

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
