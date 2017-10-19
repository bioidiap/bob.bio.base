from ..utils.processors import SequentialProcessor, ParallelProcessor
from .Extractor import Extractor
from bob.io.base import HDF5File


class MultipleExtractor(Extractor):
  """Base class for SequentialExtractor and ParallelExtractor. This class is
  not meant to be used directly."""

  def get_attributes(self, processors):
    requires_training = any(p.requires_training for p in processors)
    split_training_data_by_client = any(p.split_training_data_by_client for p
                                        in processors)
    min_extractor_file_size = min(p.min_extractor_file_size for p in
                                  processors)
    min_feature_file_size = min(p.min_feature_file_size for p in processors)
    return (requires_training, split_training_data_by_client,
            min_extractor_file_size, min_feature_file_size)

  def get_extractor_groups(self):
    groups = ['E_{}'.format(i + 1) for i in range(len(self.processors))]
    return groups

  def train_one(self, e, training_data, extractor_file, apply=False):
    if not e.requires_training:
      return
    # if any of the extractors require splitting the data, the
    # split_training_data_by_client is True.
    if e.split_training_data_by_client:
      e.train(training_data, extractor_file)
      if not apply:
        return
      training_data = [[e(d) for d in datalist] for datalist in training_data]
    # when no extractor needs splitting
    elif not self.split_training_data_by_client:
      e.train(training_data, extractor_file)
      if not apply:
        return
      training_data = [e(d) for d in training_data]
    # when e here wants it flat but the data is split
    else:
      # make training_data flat
      training_data_len = [len(datalist) for datalist in training_data]
      training_data = [d for datalist in training_data for d in datalist]
      e.train(training_data, extractor_file)
      if not apply:
        return
      # split training data
      new_training_data, i = [], 0
      for length in training_data_len:
        class_data = []
        for _ in range(length):
          class_data.append(e(training_data[i]))
          i += 1
        new_training_data.append(class_data)
      training_data = new_training_data
    return training_data

  def load(self, extractor_file):
    with HDF5File(extractor_file) as f:
      groups = self.get_extractor_groups()
      for e, group in zip(self.processors, groups):
        f.cd(group)
        e.load(f)
        f.cd('..')


class SequentialExtractor(SequentialProcessor, MultipleExtractor):
  __doc__ = SequentialProcessor.__doc__

  def __init__(self, processors):

    (requires_training, split_training_data_by_client,
     min_extractor_file_size, min_feature_file_size) = \
        self.get_attributes(processors)

    super(SequentialExtractor, self).__init__(
        processors=processors,
        requires_training=requires_training,
        split_training_data_by_client=split_training_data_by_client,
        min_extractor_file_size=min_extractor_file_size,
        min_feature_file_size=min_feature_file_size)

  def train(self, training_data, extractor_file):
    with HDF5File(extractor_file, 'w') as f:
      groups = self.get_extractor_groups()
      for e, group in zip(self.processors, groups):
        f.create_group(group)
        f.cd(group)
        training_data = self.train_one(e, training_data, f, apply=True)
        f.cd('..')


class ParallelExtractor(ParallelProcessor, MultipleExtractor):
  __doc__ = ParallelProcessor.__doc__

  def __init__(self, processors):

    (requires_training, split_training_data_by_client,
     min_extractor_file_size, min_feature_file_size) = self.get_attributes(
        processors)

    super(ParallelExtractor, self).__init__(
        processors=processors,
        requires_training=requires_training,
        split_training_data_by_client=split_training_data_by_client,
        min_extractor_file_size=min_extractor_file_size,
        min_feature_file_size=min_feature_file_size)

  def train(self, training_data, extractor_file):
    with HDF5File(extractor_file, 'w') as f:
      groups = self.get_extractor_groups()
      for e, group in zip(self.processors, groups):
        f.create_group(group)
        f.cd(group)
        self.train_one(e, training_data, f, apply=False)
        f.cd('..')


class CallableExtractor(Extractor):
  """A simple extractor that takes a callable and applies that callable to the
  input.

  Attributes
  ----------
  callable : object
      Anything that is callable. It will be used as an extractor in
      bob.bio.base.
  """

  def __init__(self, callable, **kwargs):
    super(CallableExtractor, self).__init__(**kwargs)
    self.callable = callable

  def __call__(self, data):
    return self.callable(data)
