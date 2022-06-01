import numpy as np

from scipy.spatial.distance import cdist

from ..pipelines import BioAlgorithm


class Distance(BioAlgorithm):
    """A distance algorithm to compare feature vectors.
    Many biometric algorithms are based on comparing feature vectors that
    are usually extracted by using deep neural networks.
    The most common distance function is the cosine similarity, which is
    the default in this class.
    """

    def __init__(
        self,
        distance_function="cosine",
        factor=-1,
        average_on_enroll=True,
        average_probes=False,
        probes_score_fusion="max",
        enrolls_score_fusion="max",
        **kwargs,
    ):
        """
        Parameters
        ----------
        distance_function : str or :py:class:`function`, optional
            function to be used to measure the distance of probe and model
            features compatible with :any:`scipy.spatial.distance.cdist`. If the
            function exists in scipy.spatial.distance, provide its string name
            as scipy will run an optimized version.

        factor : float
            A coefficient which is multiplied to distance (after
            distance_function) to find score between probe and model features.
            In bob.bio.base, the scores should be similarity scores (higher
            score for a genuine pair) so use this factor to make sure you are
            using similarity scores.

        average_on_enroll : bool
            Some database protocols contain multiple samples (e.g. face images)
            to create one enrollment template. This option is useful in case of
            those databases. If True, the algorithm will average the enroll
            features to create a single template. If False, the algorithm will
            use the enroll features as is and will compare the probe template
            against all features. The final score will be computed based on the
            ``enrolls_score_fusion`` option.

        average_probes : bool
            Some database protocols contain multiple samples (e.g. face images)
            to create one probe template. This option is useful in case of those
            databases. If True, the algorithm will average the probe features to
            create a single template. If False, the algorithm will use the probe
            features as is and will compare the enrollment template against all
            features. The final score will be computed based on the
            ``probes_score_fusion`` option.

        probes_score_fusion : str
            How to fuse the scores of the probes if average_probes is False and
            the database contains multiple probe samples.

        enrolls_score_fusion : str
            How to fuse the scores of the enrolls if average_on_enroll is False
            and the database contains multiple enroll samples.
        """
        super().__init__(
            probes_score_fusion=probes_score_fusion,
            enrolls_score_fusion=enrolls_score_fusion,
            **kwargs,
        )
        self.distance_function = distance_function
        self.factor = factor
        self.average_on_enroll = average_on_enroll
        self.average_probes = average_probes

    def create_templates(self, list_of_feature_sets, enroll):
        """Creates templates from the given feature sets.
        Will make sure the features are 2 dimensional before creating templates.
        Will average features over samples if ``average_on_enroll`` is True or
        ``average_probes`` is True.
        """
        list_of_feature_sets = [
            self._make_2d(data) for data in list_of_feature_sets
        ]
        # shape of list_of_feature_sets is Nx?xD
        if (enroll and self.average_on_enroll) or (
            not enroll and self.average_probes
        ):
            # we cannot call np.mean(list_of_feature_sets, axis=1) because the size of
            # axis 1 is diffent for each feature set.
            # output will be NxD
            return np.array(
                [np.mean(feat, axis=0) for feat in list_of_feature_sets]
            )
        # output shape is Nx?xD
        return list_of_feature_sets

    def _make_2d(self, X):
        """Makes sure that the features are 2 dimensional before creating enroll
        and probe templates.

        For instance, when the source is `VideoLikeContainer` the input of
        ``create_templates`` is [`VideoLikeContainer`, ....]. The concatenation
        of them makes and array of `ZxNxD`. Hence we need to stack them in `Z`.
        """
        if not len(X):
            return [[]]
        if X[0].ndim == 2:
            X = np.vstack(X)
        return np.atleast_2d(X)

    def compare(self, enroll_templates, probe_templates):
        """Compares the probe templates to the enroll templates.

        Depending on the ``average_on_enroll`` and ``average_probes`` options,
        the templates have different shapes.
        """
        # returns scores NxM where N is the number of enroll templates and M is
        # the number of probe templates
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
