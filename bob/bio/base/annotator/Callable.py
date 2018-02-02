from .Base import Base


class Callable(Base):
    """A class that wraps a callable object that annotates a sample into a
    bob.bio.annotator object."""

    def __init__(self, callable, **kwargs):
        super(Callable, self).__init__(**kwargs)
        self.callable = callable

    def annotate(self, sample, **kwargs):
        return self.callable(sample, **kwargs)
