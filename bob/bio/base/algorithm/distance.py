import numpy as np

from scipy.spatial.distance import cdist

from ..pipelines import BioAlgorithm


class Distance(BioAlgorithm):
    def __init__(
        self,
        distance_function="cosine",
        factor=-1,
        average_on_enroll=True,
        average_probes=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        distance_function : :py:class:`function`, optional
            function to be used to measure the distance of probe and model
            features.

        factor : float
            a coefficient which is multiplied to distance (after
            distance_function) to find score between probe and model features.

        average_on_enroll : bool
            [this flag is useful when there are multiple samples to be enrolled
            for each user] If True, the average of (multiple available) features
            of each user is calculated in the enrollment, and then the score for
            each probe is calculated comparing "probe" vs "average of reference
            features". If False, all the available reference features are
            enrolled for the user, and then the score for each probe is
            calculated using the average of scores (score of comparing "probe"
            vs each of the multiple available reference features for the
            enrolled user).
        average_probes : bool
            [this flag is useful when there are multiple samples to be used for
            one probe] If True, the average of (multiple available) features of
            each probe is calculated in the scoring, and then the score
        """
        super().__init__(**kwargs)
        self.distance_function = distance_function
        self.factor = factor
        self.average_on_enroll = average_on_enroll
        self.average_probes = average_probes

    def create_templates(self, list_of_feature_sets, enroll):
        list_of_feature_sets = [
            self._make_2d(data) for data in list_of_feature_sets
        ]
        if (enroll and self.average_on_enroll) or (
            not enroll and self.average_probes
        ):
            # we cannot call np.mean(list_of_feature_sets, axis=1) because the size of
            # axis 1 is diffent for each feature set.
            # output will be NxD
            return np.array(
                [np.mean(feat, axis=0) for feat in list_of_feature_sets]
            )
        # output Nx?xD
        return list_of_feature_sets

    def _make_2d(self, X):
        """
        This function will make sure that the inputs are ndim=2 before enrollment and scoring.

        For instance, when the source is `VideoLikeContainer` the input of `enroll:enroll_features` and  `score:probes` are
        [`VideoLikeContainer`, ....].
        The concatenation of them makes and array of `ZxNxD`. Hence we need to stack them in `Z`.
        """
        if not len(X):
            return [[]]
        if X[0].ndim == 2:
            X = np.vstack(X)
        return np.atleast_2d(X)

    def compare(self, enroll_templates, probe_templates):
        # returns scores NxM where N is the number of enroll templates and M is the number of probe templates
        if self.average_on_enroll and self.average_probes:
            # enroll_templates is NxD
            enroll_templates = np.asarray(enroll_templates)
            # probe_templates is MxD
            probe_templates = np.asarray(probe_templates)
            return self.factor * cdist(
                enroll_templates, probe_templates, self.distance_function
            )
        elif self.average_on_enroll:
            # enroll_templates is NxD
            enroll_templates = np.asarray(enroll_templates)
            # probe_templates is Mx?xD
            scores = []
            for probe in probe_templates:
                s = self.factor * cdist(
                    enroll_templates, probe, self.distance_function
                )
                # s is Nx?, we want s to be N
                s = self.fuse_probe_scores(s, axis=1)
                scores.append(s)
            return np.array(scores).T
        elif self.average_probes:
            # enroll_templates is Nx?xD
            # probe_templates is MxD
            probe_templates = np.asarray(probe_templates)
            scores = []
            for enroll in enroll_templates:
                s = self.factor * cdist(
                    enroll, probe_templates, self.distance_function
                )
                # s is ?xM, we want s to be M
                s = self.fuse_enroll_scores(s, axis=0)
                scores.append(s)
            return np.array(scores)
        else:
            # enroll_templates is Nx?1xD
            # probe_templates is Mx?2xD
            scores = []
            for enroll in enroll_templates:
                scores.append([])
                for probe in probe_templates:
                    s = self.factor * cdist(
                        enroll, probe, self.distance_function
                    )
                    # s is ?1x?2, we want s to be scalar
                    s = self.fuse_probe_scores(s, axis=1)
                    s = self.fuse_enroll_scores(s, axis=0)
                    scores[-1].append(s)
            return np.array(scores)
