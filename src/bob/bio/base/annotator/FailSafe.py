import logging

from .. import load_resource
from . import Annotator

logger = logging.getLogger(__name__)


class FailSafe(Annotator):
    """A fail-safe annotator.
    This annotator takes a list of annotator and tries them until you get your
    annotations.
    The annotations of previous annotator is passed to the next one.

    Attributes
    ----------
    annotators : list
        A list of annotators to try
    required_keys : list
        A list of keys that should be available in annotations to stop trying
        different annotators.
    only_required_keys : bool
        If True, the annotations will only contain the ``required_keys``.
    """

    def __init__(
        self, annotators, required_keys, only_required_keys=False, **kwargs
    ):
        super(FailSafe, self).__init__(**kwargs)
        self.annotators = []
        for annotator in annotators:
            if isinstance(annotator, str):
                annotator = load_resource(annotator, "annotator")
            self.annotators.append(annotator)
        self.required_keys = list(required_keys)
        self.only_required_keys = only_required_keys

    def annotate(self, sample, **kwargs):
        if "annotations" not in kwargs or kwargs["annotations"] is None:
            kwargs["annotations"] = {}
        for annotator in self.annotators:
            try:
                annotations = annotator.transform(
                    [sample], **{k: [v] for k, v in kwargs.items()}
                )[0]
            except Exception:
                logger.debug(
                    "The annotator `%s' failed to annotate!",
                    annotator,
                    exc_info=True,
                )
                annotations = None
            if not annotations:
                logger.debug(
                    "Annotator `%s' returned empty annotations.", annotator
                )
            else:
                logger.debug("Annotator `%s' succeeded!", annotator)
            kwargs["annotations"].update(annotations or {})
            # check if we have all the required annotations
            if all(key in kwargs["annotations"] for key in self.required_keys):
                break
        else:  # this else is for the for loop
            # we don't want to return half of the annotations
            kwargs["annotations"] = None
        if self.only_required_keys:
            for key in list(kwargs["annotations"].keys()):
                if key not in self.required_keys:
                    del kwargs["annotations"][key]
        return kwargs["annotations"]

    def transform(self, samples, **kwargs):
        """
        Takes a batch of data and tries annotating them while unsuccessful.

        Tries each annotator given at the creation of FailSafe when the previous
        one fails.

        Each ``kwargs`` value is a list of parameters, with each element of those
        lists corresponding to each element of ``sample_batch`` (for example:
        with ``[s1, s2, ...]`` as ``samples_batch``, ``kwargs['annotations']``
        should contain ``[{<s1_annotations>}, {<s2_annotations>}, ...]``).
        """
        kwargs = translate_kwargs(kwargs, len(samples))
        return [
            self.annotate(sample, **kw) for sample, kw in zip(samples, kwargs)
        ]


def translate_kwargs(kwargs, size):
    new_kwargs = [{}] * size

    if not kwargs:
        return new_kwargs

    for k, value_list in kwargs.items():
        if len(value_list) != size:
            raise ValueError(
                f"Got {value_list} in kwargs which is not of the same length of samples {size}"
            )
        for kw, v in zip(new_kwargs, value_list):
            kw[k] = v

    return new_kwargs
