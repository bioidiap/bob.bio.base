import numpy


class SequentialProcessor(object):
  """A helper class which takes several processors and applies them one by one
  sequentially

  Attributes
  ----------
  processors : list
      A list of processors to apply.
  """

  def __init__(self, processors):
    super(SequentialProcessor, self).__init__()
    self.processors = processors

  def __call__(self, data, **kwargs):
    """Applies the processors on the data sequentially. The output of the first
    one goes as input to the next one.

    Parameters
    ----------
    data : object
        The data that needs to be processed.
    **kwargs
        Any kwargs are passed to the processors.

    Returns
    -------
    object
        The processed data.
    """
    for processor in self.processors:
      try:
        data = processor(data, **kwargs)
      except ValueError:
        data = processor(data)
    return data


class ParallelProcessor(object):
  """A helper class which takes several processors and applies them on each
  processor separately and outputs a list of their outputs in the end.

  Attributes
  ----------
  processors : list
      A list of processors to apply.
  stack : bool
      If True (default), :any:`numpy.hstack` is called on the list of outputs.
  """

  def __init__(self, processors, stack=True):
    super(ParallelProcessor, self).__init__()
    self.processors = processors
    self.stack = stack

  def __call__(self, data, **kwargs):
    """Applies the processors on the data independently and outputs a list of
    their outputs.

    Parameters
    ----------
    data : object
        The data that needs to be processed.
    **kwargs
        Any kwargs are passed to the processors.

    Returns
    -------
    object
        The processed data.
    """
    output = []
    for processor in self.processors:
      try:
        out = processor(data, **kwargs)
      except ValueError:
        out = processor(data)
      output.append(out)
    if self.stack:
      output = numpy.hstack(output)
    return output
