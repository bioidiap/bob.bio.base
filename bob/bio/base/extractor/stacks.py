from bob.extension.processors import SequentialProcessor, ParallelProcessor
from .Extractor import Extractor
from bob.io.base import HDF5File


class MultipleExtractor(Extractor):
    """Base class for SequentialExtractor and ParallelExtractor. This class is
    not meant to be used directly."""

    @staticmethod
    def get_attributes(processors):
        requires_training = any(p.requires_training for p in processors)
        split_training_data_by_client = any(p.split_training_data_by_client for
                                            p in processors)
        min_extractor_file_size = min(p.min_extractor_file_size for p in
                                      processors)
        min_feature_file_size = min(p.min_feature_file_size for p in
                                    processors)
        return (requires_training, split_training_data_by_client,
                min_extractor_file_size, min_feature_file_size)

    def get_extractor_groups(self):
        groups = ['E_{}'.format(i + 1) for i in range(len(self.processors))]
        return groups

    def train_one(self, e, training_data, extractor_file, apply=False):
        """Trains one extractor and optionally applies the extractor on the
        training data after training.

        Parameters
        ----------
        e : :any:`Extractor`
            The extractor to train. The extractor should be able to save itself
            in an opened hdf5 file.
        training_data : [object] or [[object]]
            The data to be used for training.
        extractor_file : :any:`bob.io.base.HDF5File`
            The opened hdf5 file to save the trained extractor inside.
        apply : :obj:`bool`, optional
            If ``True``, the extractor is applied to the training data after it
            is trained and the data is returned.

        Returns
        -------
        None or [object] or [[object]]
            Returns ``None`` if ``apply`` is ``False``. Otherwise, returns the
            transformed ``training_data``.
        """
        if not e.requires_training:
            # do nothing since e does not require training!
            pass
        # if any of the extractors require splitting the data, the
        # split_training_data_by_client is True.
        elif e.split_training_data_by_client:
            e.train(training_data, extractor_file)
        # when no extractor needs splitting
        elif not self.split_training_data_by_client:
            e.train(training_data, extractor_file)
        # when e here wants it flat but the data is split
        else:
            # make training_data flat
            flat_training_data = [d for datalist in training_data for d in
                                  datalist]
            e.train(flat_training_data, extractor_file)

        if not apply:
            return

        # prepare the training data for the next extractor
        if self.split_training_data_by_client:
            training_data = [[e(d) for d in datalist]
                             for datalist in training_data]
        else:
            training_data = [e(d) for d in training_data]
        return training_data

    def load(self, extractor_file):
        if not self.requires_training:
            return
        with HDF5File(extractor_file) as f:
            groups = self.get_extractor_groups()
            for e, group in zip(self.processors, groups):
                f.cd(group)
                e.load(f)
                f.cd('..')


class SequentialExtractor(SequentialProcessor, MultipleExtractor):
    """A helper class which takes several extractors and applies them one by
    one sequentially.

    Attributes
    ----------
    processors : list
        A list of extractors to apply.

    Examples
    --------
    You can use this class to apply a chain of extractors on your data. For
    example:

    >>> import numpy as np
    >>> from functools import  partial
    >>> from bob.bio.base.extractor import SequentialExtractor, CallableExtractor
    >>> raw_data = np.array([[1, 2, 3], [1, 2, 3]])
    >>> seq_extractor = SequentialExtractor(
    ...     [CallableExtractor(f) for f in
    ...      [np.cast['float64'], lambda x: x / 2, partial(np.mean, axis=1)]])
    >>> np.allclose(seq_extractor(raw_data),[ 1.,  1.])
    True
    >>> np.all(seq_extractor(raw_data) ==
    ...        np.mean(np.cast['float64'](raw_data) / 2, axis=1))
    True
    """

    def __init__(self, processors, **kwargs):

        (requires_training, split_training_data_by_client,
         min_extractor_file_size, min_feature_file_size) = \
            self.get_attributes(processors)

        super(SequentialExtractor, self).__init__(
            processors=processors,
            requires_training=requires_training,
            split_training_data_by_client=split_training_data_by_client,
            min_extractor_file_size=min_extractor_file_size,
            min_feature_file_size=min_feature_file_size,
            **kwargs)

    def train(self, training_data, extractor_file):
        with HDF5File(extractor_file, 'w') as f:
            groups = self.get_extractor_groups()
            for i, (e, group) in enumerate(zip(self.processors, groups)):
                apply = i != len(self.processors) - 1
                f.create_group(group)
                f.cd(group)
                training_data = self.train_one(e, training_data, f,
                                               apply=apply)
                f.cd('..')

    def read_feature(self, feature_file):
        return self.processors[-1].read_feature(feature_file)

    def write_feature(self, feature, feature_file):
        self.processors[-1].write_feature(feature, feature_file)


class ParallelExtractor(ParallelProcessor, MultipleExtractor):
    """A helper class which takes several extractors and applies them on
    each processor separately and yields their outputs one by one.

    Attributes
    ----------
    processors : list
        A list of extractors to apply.

    Examples
    --------
    You can use this class to apply several extractors on your data and get
    all the results back. For example:

    >>> import numpy as np
    >>> from functools import  partial
    >>> from bob.bio.base.extractor import ParallelExtractor, CallableExtractor
    >>> raw_data = np.array([[1, 2, 3], [1, 2, 3]])
    >>> parallel_extractor = ParallelExtractor(
    ...     [CallableExtractor(f) for f in
    ...      [np.cast['float64'], lambda x: x / 2.0]])
    >>> np.allclose(list(parallel_extractor(raw_data)),[[[ 1.,  2.,  3.],[ 1.,  2.,  3.]], [[ 0.5,  1. ,  1.5],[ 0.5,  1. ,  1.5]]])
    True

    The data may be further processed using a :any:`SequentialExtractor`:

    >>> from bob.bio.base.extractor import SequentialExtractor
    >>> total_extractor = SequentialExtractor(
    ...     [parallel_extractor, CallableExtractor(list),
    ...      CallableExtractor(partial(np.concatenate, axis=1))])
    >>> np.allclose(total_extractor(raw_data),[[ 1. ,  2. ,  3. ,  0.5,  1. ,  1.5],[ 1. ,  2. ,  3. ,  0.5,  1. ,  1.5]])
    True
    
    """

    def __init__(self, processors, **kwargs):

        (requires_training, split_training_data_by_client,
         min_extractor_file_size, min_feature_file_size) = self.get_attributes(
            processors)

        super(ParallelExtractor, self).__init__(
            processors=processors,
            requires_training=requires_training,
            split_training_data_by_client=split_training_data_by_client,
            min_extractor_file_size=min_extractor_file_size,
            min_feature_file_size=min_feature_file_size,
            **kwargs)

    def train(self, training_data, extractor_file):
        with HDF5File(extractor_file, 'w') as f:
            groups = self.get_extractor_groups()
            for e, group in zip(self.processors, groups):
                f.create_group(group)
                f.cd(group)
                self.train_one(e, training_data, f, apply=False)
                f.cd('..')


class CallableExtractor(Extractor):
    """A simple extractor that takes a callable and applies that callable to
    the input.

    Attributes
    ----------
    callable : object
        Anything that is callable. It will be used as an extractor in
        bob.bio.base.
    read_feature : object
        A callable object with the signature of
        ``feature = read_feature(feature_file)``. If not provided, the default
        implementation handles numpy arrays.
    write_feature : object
        A callable object with the signature of
        ``write_feature(feature, feature_file)``. If not provided, the default
        implementation handles numpy arrays.

    Examples
    --------
    You can take any function like ``numpy.cast['float32']`` to cast your data
    to float32 for example. This is useful when you want to stack several
    extractors using the :any:`SequentialExtractor` and
    :any:`ParallelExtractor` classes.
    """

    def __init__(self, callable, write_feature=None, read_feature=None,
                 **kwargs):
        super(CallableExtractor, self).__init__(**kwargs)
        self.callable = callable
        if write_feature is not None:
            self.write_feature = write_feature
        if read_feature is not None:
            self.read_feature = read_feature

    def __call__(self, data):
        return self.callable(data)
