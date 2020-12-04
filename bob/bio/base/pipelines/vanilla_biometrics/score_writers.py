#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import os
from bob.pipelines import SampleSet, DelayedSample
from bob.pipelines.sample import SAMPLE_DATA_ATTRS
from .abstract_classes import ScoreWriter
import functools
import csv
import uuid
import shutil


class FourColumnsScoreWriter(ScoreWriter):
    """
    Read and write scores using the four columns format
    :py:func:`bob.bio.base.score.load.four_column`
    """

    def write(self, probe_sampleset):
        """
        Write scores and returns a :py:class:`bob.pipelines.DelayedSample`
        containing the instruction to open the score file
        """

        def _write(probe_sampleset):
            os.makedirs(self.path, exist_ok=True)
            n_lines = 0
            filename = os.path.join(self.path, str(uuid.uuid4()) + ".txt")
            with open(filename, "w") as f:
                for probe in probe_sampleset:

                    # If it's delayed, load it
                    if isinstance(probe[0], DelayedSample):
                        probe.samples = probe.samples[0].data

                    lines = [
                        "{0} {1} {2} {3}\n".format(
                            biometric_reference.reference_id,
                            probe.reference_id,
                            probe.key,
                            biometric_reference.data,
                        )
                        for biometric_reference in probe
                    ]
                    n_lines += len(probe)
                    f.writelines(lines)
            return [filename]

        import dask.bag
        import dask

        if isinstance(probe_sampleset, dask.bag.Bag):
            return probe_sampleset.map_partitions(_write)
        return _write(probe_sampleset)


class CSVScoreWriter(ScoreWriter):
    """
    Read and write scores in CSV format, shipping all metadata with the scores

    Parameters
    ----------

    path: str
        Directory to save the scores

    n_sample_sets: int
        Number of samplesets in one chunk of scores

    exclude_list: list
        List of metadata to exclude from the CSV file

    """

    def __init__(
        self,
        path,
        n_sample_sets=1000,
        exclude_list=tuple(SAMPLE_DATA_ATTRS) + ("key", "references", "annotations"),
    ):
        super().__init__(path)
        self.n_sample_sets = n_sample_sets
        self.exclude_list = exclude_list

    def write(self, probe_sampleset):
        """
        Write scores and returns a :py:class:`bob.pipelines.DelayedSample` containing
        the instruction to open the score file
        """

        def create_csv_header(probe_sampleset):
            first_biometric_reference = probe_sampleset[0]

            probe_dict = dict(
                (k, f"probe_{k}")
                for k in probe_sampleset.__dict__.keys()
                if not (k in self.exclude_list or k.startswith("_"))
            )

            bioref_dict = dict(
                (k, f"bio_ref_{k}")
                for k in first_biometric_reference.__dict__.keys()
                if not (k in self.exclude_list or k.startswith("_"))
            )

            header = (
                ["probe_key"]
                + [probe_dict[k] for k in probe_dict]
                + [bioref_dict[k] for k in bioref_dict]
                + ["score"]
            )
            return header, probe_dict, bioref_dict

        os.makedirs(self.path, exist_ok=True)
        header, probe_dict, bioref_dict = create_csv_header(probe_sampleset[0])

        f = None
        filename = os.path.join(self.path, str(uuid.uuid4()))
        filenames = []
        for i, probe in enumerate(probe_sampleset):
            if i % self.n_sample_sets == 0:
                filename = filename + "_" + f"chunk_{i}.csv"
                filenames.append(filename)
                if f is not None:
                    f.close()
                    del f

                f = open(filename, "w")
                csv_writer = csv.writer(f)
                if i == 0:
                    csv_writer.writerow(header)

            rows = []
            probe_row = [str(probe.key)] + [
                str(getattr(probe, k)) for k in probe_dict.keys()
            ]

            # If it's delayed, load it
            if isinstance(probe[0], DelayedSample):
                probe.samples = probe.samples[0].data

            for biometric_reference in probe:
                bio_ref_row = [
                    str(getattr(biometric_reference, k))
                    for k in list(bioref_dict.keys()) + ["data"]
                ]

                rows.append(probe_row + bio_ref_row)

            csv_writer.writerows(rows)
        f.close()
        return filenames

    def post_process(self, score_paths, path):
        """
        Removing the HEADER of all files
        but the first
        """

        def _post_process(score_paths, path):
            post_process_scores = []
            os.makedirs(path, exist_ok=True)
            for i, score in enumerate(score_paths):
                fname = os.path.join(
                    path, os.path.basename(score) + "_post_process.csv"
                )
                post_process_scores.append(fname)
                if i == 0:
                    shutil.move(score, fname)
                    continue

                # Not memory intensive score writing
                with open(score, "r") as f:
                    with open(fname, "w") as f1:
                        f.readline()  # skip header line
                        for line in f:
                            f1.write(line)

                open(fname, "w").writelines(open(score, "r").readlines()[1:])
                os.remove(score)
            return post_process_scores

        import dask.bag
        import dask

        if isinstance(score_paths, dask.bag.Bag):
            all_paths = dask.delayed(list)(score_paths)
            return dask.delayed(_post_process)(all_paths, path)
        return _post_process(score_paths, path)
