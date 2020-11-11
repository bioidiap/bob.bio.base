import logging
import six
from . import Annotator
from .. import load_resource

logger = logging.getLogger(__name__)

def _isolate_kwargs(kwargs_dict, index):
    """
    Returns the kwargs to pass down to an annotator.

    Each annotator is expecting a batch of samples and the corresponding kwargs.
    """
    return {k:[v[index]] for k,v in kwargs_dict.items()}

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

    def __init__(self, annotators, required_keys, only_required_keys=False,
                 **kwargs):
        super(FailSafe, self).__init__(**kwargs)
        self.annotators = []
        for annotator in annotators:
            if isinstance(annotator, six.string_types):
                annotator = load_resource(annotator, 'annotator')
            self.annotators.append(annotator)
        self.required_keys = list(required_keys)
        self.only_required_keys = only_required_keys

    def transform(self, sample_batch, **kwargs):
        """
        Takes a batch of data and tries annotating them while unsuccessful.

        Tries each annotator given at the creation of FailSafe when the previous
        one fails.

        Each ``kwargs`` value is a list of parameters, with each element of those
        lists corresponding to each element of ``sample_batch`` (for example:
        with ``[s1, s2, ...]`` as ``samples_batch``, ``kwargs['annotations']``
        should contain ``[{<s1_annotations>}, {<s2_annotations>}, ...]``).
        """
        if 'annotations' not in kwargs or kwargs['annotations'] is None:
            kwargs['annotations'] = [{}] * len(sample_batch)
        # Executes on each sample and corresponding existing annotations
        for index, (sample, annotations) in enumerate(zip(sample_batch, kwargs['annotations'])):
            for annotator in self.annotators:
                try:
                    annot = annotator([sample], **_isolate_kwargs(kwargs, index))[0]
                except Exception:
                    logger.debug(
                        "The annotator `%s' failed to annotate!", annotator,
                        exc_info=True)
                    annot = None
                if not annot:
                    logger.debug(
                        "Annotator `%s' returned empty annotations.", annotator)
                else:
                    logger.debug("Annotator `%s' succeeded!", annotator)
                annotations.update(annot or {})
                # check if we have all the required annotations
                if all(key in annotations for key in self.required_keys):
                    break
            else:  # this else is for the for loop
                # we don't want to return half of the annotations
                annotations = None
            if self.only_required_keys:
                for key in list(annotations.keys()):
                    if key not in self.required_keys:
                        del annotations[key]
        return kwargs['annotations']
