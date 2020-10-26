import numpy
from bob.pipelines.sample import SampleSet, Sample
from bob.bio.base.pipelines.vanilla_biometrics.abstract_classes import Database

TRAIN_SUBJECTS = ["00", "50"]

# 4 subject are defined (subject_id)
SUBJECTS = ["10", "20", "30", "40"]

# Each subject has 2 samples used for enrollment (reference sample_id)
REFERENCES_SAMPLES = [1, 2]

# Each subject has 3 samples used for scoring (probe sample_id)
PROBES_SAMPLES = [4, 5, 6]

# Each sample is a 5x5 array (filled with "subject_id + sample_id")
SAMPLE_SIZE = [5,5]

# Create a simple custom database implementation
class CustomDatabase(Database):

    allow_scoring_with_all_biometric_references = False

    def background_model_samples(self):
        train_samples = []
        for subj in TRAIN_SUBJECTS:
            train_samples.append(Sample(numpy.full(SAMPLE_SIZE, int(subj)), subject=subj))
        return train_samples

    def references(self, group="dev"):
        references = []
        for subj in SUBJECTS:
            references.append(SampleSet(samples=[], subject=subj))
            for s in REFERENCES_SAMPLES:
                references[-1].insert(-1,
                    Sample(
                        data=numpy.full(SAMPLE_SIZE, int(subj)+s),
                        # key=int(subj)+s,
                        subject=subj,
                    )
                )
        return references

    def probes(self, group="dev"):
        probes = []
        for subj in SUBJECTS:
            for s in PROBES_SAMPLES:
                current_sampleset = SampleSet(
                    samples=[
                        Sample( numpy.full(SAMPLE_SIZE, int(subj)+s) )
                    ],
                    subject=subj,
                    key=s,
                    references=SUBJECTS,
                )
                probes.append(current_sampleset)
        return probes

# Instantiate 'database' for vanilla-biometrics
database = CustomDatabase()
