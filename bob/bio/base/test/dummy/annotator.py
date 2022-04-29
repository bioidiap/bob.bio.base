from random import random

from bob.bio.base.annotator import Callable, FailSafe


def simple_annotator(image_batch, **kwargs):
    all_annotations = []
    for image in image_batch:
        all_annotations.append(
            {
                "topleft": (0, 0),
                "bottomright": image.shape,
            }
        )
    return all_annotations


def moody_annotator(image_batch, **kwargs):
    all_annotations = simple_annotator(image_batch, **kwargs)
    for annot in all_annotations:
        if random() < 0.5:
            del annot["bottomright"]
    return all_annotations


def fail_annotator(image_batch, **kwargs):
    all_annotations = []
    for image in image_batch:
        all_annotations.append({})
    return all_annotations


annotator = FailSafe(
    [Callable(fail_annotator), Callable(simple_annotator)],
    required_keys=["topleft", "bottomright"],
)
