#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import os
from bob.pipelines import SampleSet, DelayedSample
from .abstract_classes import ScoreWriter
import functools
import csv


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
            filename = os.path.join(path, str(probe.subject)) + ".txt"
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


class CSVScoreWriter(ScoreWriter):
    """
    Read and write scores in CSV format, shipping all metadata with the scores    

    Parameters
    ----------

    n_sample_sets: 
        Number of samplesets in one chunk

    """

    def __init__(self, n_sample_sets=1000):
        self.n_sample_sets = n_sample_sets

    def write(self, probe_sampleset, path):
        """
        Write scores and returns a :any:`bob.pipelines.DelayedSample` containing
        the instruction to open the score file
        """

        exclude_list = ["samples", "key", "data", "load", "_data", "references"]

        def create_csv_header(probe_sampleset):
            first_biometric_reference = probe_sampleset[0]

            probe_dict = dict(
                (k, f"probe_{k}")
                for k in probe_sampleset.__dict__.keys()
                if k not in exclude_list
            )

            bioref_dict = dict(
                (k, f"bio_ref_{k}")
                for k in first_biometric_reference.__dict__.keys()
                if k not in exclude_list
            )

            header = (
                ["probe_key"]
                + [probe_dict[k] for k in probe_dict]
                + [bioref_dict[k] for k in bioref_dict]
                + ["score"]
            )
            return header, probe_dict, bioref_dict

        os.makedirs(path, exist_ok=True)
        checkpointed_scores = []

        header, probe_dict, bioref_dict = create_csv_header(probe_sampleset[0])

        for probe in probe_sampleset:
            filename = os.path.join(path, str(probe.subject)) + ".csv"
            with open(filename, "w") as f:

                csv_write = csv.writer(f)
                csv_write.writerow(header)

                rows = []
                probe_row = [str(probe.key)] + [
                    str(probe.__dict__[k]) for k in probe_dict.keys()
                ]

                for biometric_reference in probe:
                    bio_ref_row = [
                        str(biometric_reference.__dict__[k])
                        for k in list(bioref_dict.keys()) + ["data"]
                    ]

                    rows.append(probe_row + bio_ref_row)

                csv_write.writerows(rows)
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

        # CSV files tends to be very big
        # here, here we write them in chunks

        base_dir = os.path.splitext(filename)[0]
        os.makedirs(base_dir, exist_ok=True)
        f = None
        for i, samplesets in enumerate(samplesets_list):
            if i% self.n_sample_sets==0:
                if f is not None:
                    f.close()
                    del f

                filename = os.path.join(base_dir, f"chunk_{i}.csv")
                f = open(filename, "w")

            for sset in samplesets:
                for s in sset:
                    if i==0:
                        f.writelines(s.data)
                    else:
                        f.writelines(s.data[1:])
            samplesets_list[i] = None