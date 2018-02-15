import logging
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
    """

    def __init__(self, annotators, required_keys, **kwargs):
        super(FailSafe, self).__init__(**kwargs)
        self.annotators = list(annotators)
        self.required_keys = list(required_keys)

    def annotate(self, sample, **kwargs):
        if 'annotations' not in kwargs or kwargs['annotations'] is None:
            kwargs['annotations'] = {}
        for annotator in self.annotators:
            try:
                annotations = annotator(sample, **kwargs)
            except Exception:
                logger.debug(
                    "The annotator `%s' failed to annotate!", annotator,
                    exc_info=True)
                annotations = {}
            if not annotations:
                logger.debug(
                    "Annotator `%s' returned empty annotations.", annotator)
            kwargs['annotations'].update(annotations)
            # check if we have all the required annotations
            if all(key in kwargs['annotations'] for key in self.required_keys):
                break
        else:  # this else is for the for loop
            # we don't want to return half of the annotations
            kwargs['annotations'] = {}
        return kwargs['annotations']
