#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


import csv
import os
import uuid

from bob.pipelines import DelayedSample, SampleSet
from bob.pipelines.sample import SAMPLE_DATA_ATTRS

from .abstract_classes import ScoreWriter


class FourColumnsScoreWriter(ScoreWriter):
    """
    Read and write scores using the four columns format
    :py:func:`bob.bio.base.score.load.four_column`
    """

    def __init__(self, path, extension=".txt", **kwargs):
        super().__init__(path, extension, **kwargs)

    def write(self, probe_sampleset: list[SampleSet]):
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
                            biometric_reference.subject_id,
                            probe.subject_id,
                            probe.template_id,
                            biometric_reference.data,
                        )
                        for biometric_reference in probe
                    ]
                    n_lines += len(probe)
                    f.writelines(lines)
            return [filename]

        import dask
        import dask.bag

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

    exclude_list: list
        List of metadata to exclude from the CSV file

    """

    def __init__(
        self,
        path,
        exclude_list=tuple(SAMPLE_DATA_ATTRS)
        + ("key", "references", "annotations"),
        **kwargs,
    ):
        super().__init__(path, **kwargs)
        self.exclude_list = exclude_list

    def write(self, probe_sampleset: list[SampleSet]):
        """
        Write scores and returns a :py:class:`bob.pipelines.DelayedSample` containing
        the instruction to open the score file
        """

        def create_csv_header(probe_sampleset):
            first_biometric_reference = probe_sampleset[0]

            probe_dict = dict(
                (k, f"probe_{k}")
                for k in probe_sampleset.__dict__.keys()
                if not (
                    k in self.exclude_list
                    or k.startswith("_")
                    or k == "template_id"
                )
            )

            bioref_dict = dict(
                (k, f"bio_ref_{k}")
                for k in first_biometric_reference.__dict__.keys()
                if not (k in self.exclude_list or k.startswith("_"))
            )

            header = [
                "probe_template_id",
                *probe_dict.values(),
                *bioref_dict.values(),
                "score",
            ]
            return header, probe_dict, bioref_dict

        os.makedirs(self.path, exist_ok=True)
        header, probe_dict, bioref_dict = create_csv_header(probe_sampleset[0])

        filename = os.path.join(self.path, str(uuid.uuid4()))
        with open(filename, "w") as f:
            csv_writer = csv.writer(f)
            rows = []
            for i, probe in enumerate(probe_sampleset):
                # Writing the header
                if i == 0:
                    csv_writer.writerow(header)

                probe_row = [str(probe.template_id)] + [
                    str(getattr(probe, k)) for k in probe_dict
                ]

                # Iterating over the biometric references
                for biometric_reference in probe:
                    bio_ref_row = [
                        str(getattr(biometric_reference, k))
                        for k in list(bioref_dict.keys()) + ["data"]
                    ]

                    rows.append(probe_row + bio_ref_row)

            csv_writer.writerows(rows)
        return [filename]

    def post_process(self, score_paths, path):
        """
        Removing the HEADER of all files
        but the first
        """

        def _post_process(score_paths, path):
            post_processed_scores = []
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                for i, score in enumerate(score_paths):
                    post_processed_scores.append(score)

                    # Not memory intensive score writing
                    with open(score, "r") as f1:
                        if i > 0:
                            f1.readline()  # skip header line
                        for line in f1:
                            f.write(line)
                    os.remove(score)

            return post_processed_scores

        import dask
        import dask.bag

        if isinstance(score_paths, dask.bag.Bag):
            all_paths = dask.delayed(list)(score_paths)
            return dask.delayed(_post_process)(all_paths, path)
        return _post_process(score_paths, path)
