"""
#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

Implementation of a pipeline and an algorithm that 
computes Z, T and ZT Score Normalization of a :any:`BioAlgorithm`
"""

from bob.pipelines import DelayedSample, Sample, SampleSet
import numpy as np
import dask
import functools
import pickle
import os
from .score_writers import FourColumnsScoreWriter
import logging

logger = logging.getLogger(__name__)


class ZTNormPipeline(object):
    """
    Apply Z, T or ZT Score normalization on top of VanillaBiometric Pipeline

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf


    Example
    -------
       >>> transformer = make_pipeline([])
       >>> biometric_algorithm = Distance()
       >>> vanilla_biometrics_pipeline = VanillaBiometricsPipeline(transformer, biometric_algorithm)
       >>> zt_pipeline = ZTNormVanillaBiometricsPipeline(vanilla_biometrics_pipeline)
       >>> zt_pipeline(...)

    Parameters
    ----------

        vanilla_biometrics_pipeline: :any:`VanillaBiometricsPipeline`
          An instance :any:`VanillaBiometricsPipeline` to the wrapped with score normalization

        z_norm: bool
          If True, applies ZScore normalization on top of raw scores.

        t_norm: bool
          If True, applies TScore normalization on top of raw scores.
          If both, z_norm and t_norm are true, it applies score normalization

    """

    def __init__(
        self,
        vanilla_biometrics_pipeline,
        z_norm=True,
        t_norm=True,
        score_writer=FourColumnsScoreWriter("./scores.txt"),
    ):
        self.vanilla_biometrics_pipeline = vanilla_biometrics_pipeline
        self.ztnorm_solver = ZTNorm()

        self.z_norm = z_norm
        self.t_norm = t_norm
        self.score_writer = score_writer

        if not z_norm and not t_norm:
            raise ValueError(
                "Both z_norm and t_norm are False. No normalization will be applied"
            )

    def __call__(
        self,
        background_model_samples,
        biometric_reference_samples,
        probe_samples,
        zprobe_samples=None,
        t_biometric_reference_samples=None,
        allow_scoring_with_all_biometric_references=False,
    ):

        self.transformer = self.train_background_model(background_model_samples)

        # Create biometric samples
        biometric_references = self.create_biometric_reference(
            biometric_reference_samples
        )

        raw_scores, probe_features = self.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

        # Z NORM
        if self.z_norm:
            if zprobe_samples is None:
                raise ValueError("No samples for `z_norm` was provided")
            z_normed_scores, z_probe_features = self.compute_znorm_scores(
                zprobe_samples,
                raw_scores,
                biometric_references,
                allow_scoring_with_all_biometric_references,
            )
        if self.t_norm:
            if t_biometric_reference_samples is None:
                raise ValueError("No samples for `t_norm` was provided")
        else:
            # In case z_norm=True and t_norm=False
            return z_normed_scores

        # T NORM
        t_normed_scores, t_scores, t_biometric_references = self.compute_tnorm_scores(
            t_biometric_reference_samples,
            probe_features,
            raw_scores,
            allow_scoring_with_all_biometric_references,
        )
        if not self.z_norm:
            # In case z_norm=False and t_norm=True
            return t_normed_scores

        # ZT NORM
        zt_normed_scores = self.compute_ztnorm_scores(
            z_probe_features,
            t_biometric_references,
            z_normed_scores,
            t_scores,
            allow_scoring_with_all_biometric_references,
        )
        

        # TODO: Do the score write
        #if self.vanilla_biometrics_pipeline.score_writer is not None:
        #    return self.write_scores(scores)

        return raw_scores, z_normed_scores, t_normed_scores, zt_normed_scores

    def train_background_model(self, background_model_samples):
        return self.vanilla_biometrics_pipeline.train_background_model(
            background_model_samples
        )

    def create_biometric_reference(self, biometric_reference_samples):
        return self.vanilla_biometrics_pipeline.create_biometric_reference(
            biometric_reference_samples
        )

    def compute_scores(
        self,
        probe_samples,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        return self.vanilla_biometrics_pipeline.compute_scores(
            probe_samples,
            biometric_references,
            allow_scoring_with_all_biometric_references,
        )

    def compute_znorm_scores(
        self,
        zprobe_samples,
        probe_scores,
        biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):

        # zprobe_samples = self._inject_references(zprobe_samples, biometric_references)

        z_scores, z_probe_features = self.compute_scores(
            zprobe_samples, biometric_references
        )

        z_normed_scores = self.ztnorm_solver.compute_znorm_scores(
            probe_scores, z_scores, biometric_references
        )

        return z_normed_scores, z_probe_features

    def compute_tnorm_scores(
        self,
        t_biometric_reference_samples,
        probe_features,
        probe_scores,
        allow_scoring_with_all_biometric_references=False,
    ):

        t_biometric_references = self.create_biometric_reference(
            t_biometric_reference_samples
        )

        # probe_features = self._inject_references(probe_features, t_biometric_references)

        # Reusing the probe features
        t_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.score_samples(
            probe_features,
            t_biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        t_normed_scores = self.ztnorm_solver.compute_tnorm_scores(
            probe_scores, t_scores, t_biometric_references
        )

        return t_normed_scores, t_scores, t_biometric_references

    def compute_ztnorm_scores(
        self,
        z_probe_features,
        t_biometric_references,
        z_normed_scores,
        t_scores,
        allow_scoring_with_all_biometric_references=False,
    ):

        # z_probe_features = self._inject_references(
        #    z_probe_features, t_biometric_references
        # )

        # Reusing the zprobe_features and t_biometric_references
        zt_scores = self.vanilla_biometrics_pipeline.biometric_algorithm.score_samples(
            z_probe_features,
            t_biometric_references,
            allow_scoring_with_all_biometric_references=allow_scoring_with_all_biometric_references,
        )

        # Z Normalizing the T-normed scores
        z_normed_t_normed = self.ztnorm_solver.compute_znorm_scores(
            t_scores, zt_scores, t_biometric_references
        )

        # (Z Normalizing the T-normed scores) the Z normed scores
        zt_normed_scores = self.ztnorm_solver.compute_tnorm_scores(
            z_normed_scores, z_normed_t_normed, t_biometric_references
        )

        return zt_normed_scores

    def write_scores(self, scores):
        return self.vanilla_biometrics_pipeline.write_scores(scores)

    def post_process(self, score_paths, filename):
        return self.vanilla_biometrics_pipeline.post_process(score_paths, filename)

class ZTNorm(object):
    """
    Computes Z, T and ZT Score Normalization of a :any:`BioAlgorithm`

    Reference bibliography from: A Generative Model for Score Normalization in Speaker Recognition
    https://arxiv.org/pdf/1709.09868.pdf

    """

    def _norm(self, score, mu, std):
        # Reference: https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/master/bob/learn/em/test/test_ztnorm.py
        # Axis 0=ZNORM
        # Axi1 1=TNORM
        return (score - mu) / std

        """
        if axis == 1:
            return (
                score
                - np.tile(mu.reshape(N, 1), (1, score.shape[1]))
            ) / np.tile(std.reshape(N, 1), (1, score.shape[1]))
        else:
            return (
                score
                - np.tile(mu.reshape(1, N), (score.shape[0], 1))
            ) / np.tile(std.reshape(1, N), (score.shape[0], 1))
        """
        

    def _compute_std(self, mu, norm_base_scores, axis=1):
        # Reference: https://gitlab.idiap.ch/bob/bob.learn.em/-/blob/master/bob/learn/em/test/test_ztnorm.py
        # Axis 0=ZNORM
        # Axi1 1=TNORM
        if axis == 1:
            return np.sqrt(
                np.sum(
                    (
                        norm_base_scores
                        - np.tile(
                            mu.reshape(norm_base_scores.shape[0], 1),
                            (1, norm_base_scores.shape[1]),
                        )
                    )
                    ** 2,
                    axis=1,
                )
                / (norm_base_scores.shape[1] - 1)
            )

        else:

            return np.sqrt(
                np.sum(
                    (
                        norm_base_scores
                        - np.tile(
                            mu.reshape(1, norm_base_scores.shape[1]),
                            (norm_base_scores.shape[0], 1),
                        )
                    )
                    ** 2,
                    axis=0,
                )
                / (norm_base_scores.shape[0] - 1)
            )

    def _compute_stats(self, sampleset_for_norm, biometric_references, axis=0):
        """
        Compute statistics for Z and T Norm.

        The way the scores are organized (probe vs bioref);
        axis=0 computes CORRECTLY the statistics for ZNorm
        axis=1 computes CORRECTLY the statistics for TNorm
        """

        # Dumping all scores
        if isinstance(sampleset_for_norm[0][0], DelayedSample):
            score_floats = np.array(
                [f.data for sset in sampleset_for_norm for s in sset for f in s.data]
            )
        else:
            score_floats = np.array(
                [s.data for sset in sampleset_for_norm for s in sset]
            )

        # Reshaping in PROBE vs BIOMETRIC_REFERENCES
        n_probes = len(sampleset_for_norm)
        n_references = len(biometric_references)
        score_floats = score_floats.reshape((n_probes, n_references))

        # AXIS ON THE MODELS
        big_mu = np.mean(score_floats, axis=axis)
        #big_std = np.std(score_floats, axis=axis)
        big_std = self._compute_std(big_mu, score_floats, axis=axis)

        # Creating statistics structure with subject id as the key
        stats = {}
        if axis == 0:
            # TODO: NEED TO SOLVE THIS FETCHING.
            # IT SHOULD BE TRANSPARENT
            if isinstance(sampleset_for_norm[0][0], DelayedSample):
                sset = sampleset_for_norm[0].samples[0].data
            else:
                sset = sampleset_for_norm[0]

            for mu, std, s in zip(big_mu, big_std, sset):
                stats[s.subject] = {"big_mu": mu, "big_std": std}
        else:
            for mu, std, sset in zip(big_mu, big_std, sampleset_for_norm):
                stats[sset.subject] = {"big_mu": mu, "big_std": std}

        return stats

    def _znorm_samplesets(self, probe_scores, stats):
        # Normalizing
        # TODO: THIS TENDS TO BE EXTREMLY SLOW

        z_normed_score_samples = []
        for probe_score in probe_scores:
            z_normed_score_samples.append(self._apply_znorm(probe_score, stats))
        return z_normed_score_samples

    def _apply_znorm(self, probe_score, stats):

        z_normed_score = SampleSet([], parent=probe_score)
        for biometric_reference_score in probe_score:

            mu = stats[biometric_reference_score.subject]["big_mu"]
            std = stats[biometric_reference_score.subject]["big_std"]

            score = self._norm(biometric_reference_score.data, mu, std)
            new_sample = Sample(score, parent=biometric_reference_score)
            z_normed_score.samples.append(new_sample)
        return z_normed_score

    def _tnorm_samplesets(self, probe_scores, stats):
        # Normalizing
        # TODO: THIS TENDS TO BE EXTREMLY SLOW
        # MAYBE THIS COULD BE DELAYED OR RUN ON TOP OF

        t_normed_score_samples = []
        for probe_score in probe_scores:
            t_normed_score_samples.append(self._apply_tnorm(probe_score, stats))

        return t_normed_score_samples

    def _apply_tnorm(self, probe_score, stats):
        # Normalizing

        t_normed_scores = SampleSet([], parent=probe_score)

        mu = stats[probe_score.subject]["big_mu"]
        std = stats[probe_score.subject]["big_std"]

        for biometric_reference_score in probe_score:
            score = self._norm(biometric_reference_score.data, mu, std)
            new_sample = Sample(score, parent=biometric_reference_score)
            t_normed_scores.samples.append(new_sample)

        return t_normed_scores

    def compute_znorm_scores(
        self, probe_scores, sampleset_for_znorm, biometric_references
    ):
        """
        Base Z-normalization function
        """

        stats = self._compute_stats(sampleset_for_znorm, biometric_references, axis=0)

        return self._znorm_samplesets(probe_scores, stats)

    def compute_tnorm_scores(
        self,
        probe_scores,
        sampleset_for_tnorm,
        t_biometric_references,
        allow_scoring_with_all_biometric_references=False,
    ):
        """
        Base T-normalization function
        """

        stats = self._compute_stats(sampleset_for_tnorm, t_biometric_references, axis=1)

        return self._tnorm_samplesets(probe_scores, stats)


class ZTNormDaskWrapper(object):
    """
    Wrap :any:`ZTNorm` to work with DASK

    Parameters
    ----------

        ztnorm: :any:`ZTNorm`
            ZTNorm Pipeline
    """

    def __init__(self, ztnorm):
        self.ztnorm = ztnorm

    def compute_znorm_scores(
        self, probe_scores, sampleset_for_znorm, biometric_references
    ):

        # Reducing all the Z-Scores to compute the stats
        all_scores_for_znorm = dask.delayed(list)(sampleset_for_znorm)

        stats = dask.delayed(self.ztnorm._compute_stats)(
            all_scores_for_znorm, biometric_references, axis=0
        )

        return probe_scores.map_partitions(self.ztnorm._znorm_samplesets, stats)

    def compute_tnorm_scores(
        self, probe_scores, sampleset_for_tnorm, t_biometric_references
    ):

        # Reducing all the Z-Scores to compute the stats
        all_scores_for_tnorm = dask.delayed(list)(sampleset_for_tnorm)

        stats = dask.delayed(self.ztnorm._compute_stats)(
            all_scores_for_tnorm, t_biometric_references, axis=1
        )

        return probe_scores.map_partitions(self.ztnorm._tnorm_samplesets, stats)


class ZTNormCheckpointWrapper(object):
    """
    Wrap :any:`ZTNorm` to work with DASK

    Parameters
    ----------

        ztnorm: :any:`ZTNorm`
            ZTNorm Pipeline
    """

    def __init__(self, ztnorm, base_dir, force=False):

        if not isinstance(ztnorm, ZTNorm):
            raise ValueError("This class only wraps `ZTNorm` objects")

        self.ztnorm = ztnorm
        self.znorm_score_path = os.path.join(base_dir, "znorm_scores")
        self.tnorm_score_path = os.path.join(base_dir, "tnorm_scores")
        self.force = force
        self.base_dir = base_dir


    def _write_scores(self, samples, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").write(pickle.dumps(samples))

    def _load(self, path):
        return pickle.loads(open(path, "rb").read())

    def _apply_znorm(self, probe_score, stats):

        path = os.path.join(self.znorm_score_path, str(probe_score.key) + ".pkl")

        if self.force or not os.path.exists(path):
            z_normed_score = self.ztnorm._apply_znorm(probe_score, path)

            self.write_scores(z_normed_score.samples)

            z_normed_score = SampleSet(
                [
                    DelayedSample(
                        functools.partial(self._load, path), parent=probe_score
                    )
                ],
                parent=probe_score,
            )
        else:
            z_normed_score = SampleSet(self._load(path), parent=probe_score)

        return z_normed_score

    def compute_znorm_scores(
        self, probe_scores, sampleset_for_znorm, biometric_references
    ):

        return self.ztnorm.compute_znorm_scores(
            probe_scores, sampleset_for_znorm, biometric_references
        )

    def compute_tnorm_scores(
        self, probe_scores, sampleset_for_tnorm, t_biometric_references
    ):

        return self.ztnorm.compute_tnorm_scores(
            probe_scores, sampleset_for_tnorm, t_biometric_references
        )

    def _compute_stats(self, sampleset_for_norm, biometric_references, axis=0):
        return self.ztnorm._compute_stats(
            sampleset_for_norm, biometric_references, axis=axis
        )

    def _znorm_samplesets(self, probe_scores, stats):
        return self.ztnorm._znorm_samplesets(probe_scores, stats)

    def _tnorm_samplesets(self, probe_scores, stats):
        return self.ztnorm._tnorm_samplesets(probe_scores, stats)
