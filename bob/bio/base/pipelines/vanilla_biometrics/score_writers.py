#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import os
from bob.pipelines import SampleSet, DelayedSample
from .abstract_classes import ScoreWriter
import functools

class FourColumnsScoreWriter(ScoreWriter):
    """
    Read and write scores using the four columns format
    :any:`bob.bio.base.score.load.four_column`
    """

    def write(self, probe_sampleset, path):
        """
        Write scores and returns a :any:`bob.pipelines.DelayedSample` containing
        the instruction to open the score file
        """
        os.makedirs(path, exist_ok=True)
        checkpointed_scores = []

        for probe in probe_sampleset:

            lines = [
                "{0} {1} {2} {3}\n".format(
                    biometric_reference.subject,
                    probe.subject,
                    probe.key,
                    biometric_reference.data,
                )
                for biometric_reference in probe
            ]
            filename = os.path.join(path, probe.subject)
            open(filename, "w").writelines(lines)
            checkpointed_scores.append(
                SampleSet(
                    [
                        DelayedSample(
                            functools.partial(self.read, filename), parent=probe
                        )
                    ],
                    parent=probe,
                )
            )
        return checkpointed_scores

    def read(self, path):
        """
        Base Instruction to load a score file
        """
        return open(path).readlines()

    def concatenate_write_scores(self, samplesets_list, filename):
        """
        Given a list of samplsets, write them all in a single file
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        f = open(filename, "w")
        for samplesets in samplesets_list:
            for sset in samplesets:
                for s in sset:
                    f.writelines(s.data)