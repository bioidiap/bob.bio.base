#!/usr/bin/env python
# vim: set fileencoding=utf-8 :


"""Re-usable blocks for legacy bob.bio.base algorithms"""


import os
import copy
import functools

import bob.io.base


from .legacy import DatabaseConnector
from .blocks import SampleLoader
from bob.pipelines.sample.sample import SampleSet, DelayedSample, Sample 
from bob.pipelines.utils import is_picklable

import logging
logger = logging.getLogger("bob.bio.base")


class DatabaseConnectorAnnotated(DatabaseConnector):
    """Wraps a bob.bio.base database and generates conforming samples for datasets
    that has annotations

    This connector allows wrapping generic bob.bio.base datasets and generate
    samples that conform to the specifications of biometric pipelines defined
    in this package.


    Parameters
    ----------

    database : object
        An instantiated version of a bob.bio.base.Database object

    protocol : str
        The name of the protocol to generate samples from.
        To be plugged at :py:method:`bob.db.base.Database.objects`.

    """

    def __init__(self, database, protocol):
        super(DatabaseConnectorAnnotated, self).__init__(database, protocol)
       

    def background_model_samples(self):
        """Returns :py:class:`Sample`'s to train a background model (group
        ``world``).


        Returns
        -------

            samples : list
                List of samples conforming the pipeline API for background
                model training.  See, e.g., :py:func:`.pipelines.first`.

        """

        # TODO: This should be organized by client
        retval = []

        objects = self.database.objects(protocol=self.protocol, groups="world")

        return [
            SampleSet(
                [
                    DelayedSample(
                        load=functools.partial(
                            k.load,
                            self.database.original_directory,
                            self.database.original_extension,
                        ),
                        id=k.id,
                        path=k.path,
                        annotations=self.database.annotations(k)
                    )
                ]
            )
            for k in objects
        ]

    def references(self, group="dev"):
        """Returns :py:class:`Reference`'s to enroll biometric references


        Parameters
        ----------

            group : :py:class:`str`, optional
                A ``group`` to be plugged at
                :py:meth:`bob.db.base.Database.objects`


        Returns
        -------

            references : list
                List of samples conforming the pipeline API for the creation of
                biometric references.  See, e.g., :py:func:`.pipelines.first`.

        """

        retval = []

        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

            objects = self.database.objects(
                protocol=self.protocol,
                groups=group,
                model_ids=(m,),
                purposes="enroll",
            )

            retval.append(
                SampleSet(
                    [
                        DelayedSample(
                            load=functools.partial(
                                k.load,
                                self.database.original_directory,
                                self.database.original_extension,
                            ),
                            id=k.id,
                            path=k.path,
                            annotations=self.database.annotations(k)
                        )
                        for k in objects
                    ],
                    id=m,
                    path=str(m),
                    subject=objects[0].client_id,
                )
            )

        return retval

    def probes(self, group):
        """Returns :py:class:`Probe`'s to score biometric references


        Parameters
        ----------

            group : str
                A ``group`` to be plugged at
                :py:meth:`bob.db.base.Database.objects`


        Returns
        -------

            probes : list
                List of samples conforming the pipeline API for the creation of
                biometric probes.  See, e.g., :py:func:`.pipelines.first`.

        """

        probes = dict()

        for m in self.database.model_ids_with_protocol(protocol=self.protocol, groups=group):

            # Getting all the probe objects from a particular biometric
            # reference
            objects = self.database.objects(
                protocol=self.protocol,
                groups=group,
                model_ids=(m,),
                purposes="probe",
            )

            # Creating probe samples
            for o in objects:
                if o.id not in probes:
                    probes[o.id] = SampleSet(
                        [
                            DelayedSample(
                                load=functools.partial(
                                    o.load,
                                    self.database.original_directory,
                                    self.database.original_extension,                                    
                                ),
                                id=o.id,
                                path=o.path,
                                annotations=self.database.annotations(o)
                            )
                        ],
                        id=o.id,
                        path=o.path,
                        subject=o.client_id,
                        references=[m],
                    )
                else:
                    probes[o.id].references.append(m)
     
        return list(probes.values())


class SampleLoaderAnnotated(SampleLoader):
    """Adaptor for loading, preprocessing and feature extracting samples that uses annotations

    This adaptor class wraps around sample:

    .. code-block:: text

       [loading [-> preprocessing [-> extraction]]]

    The input sample object must obbey the following (minimal) API:

        * attribute ``id``: Contains an unique (string-fiable) identifier for
          processed samples
        * attribute ``data``: Contains the data for this sample

    Optional checkpointing is also implemented for each of the states,
    independently.  You may check-point just the preprocessing, feature
    extraction or both.


    Parameters
    ----------

    pipeline : :py:class:`list` of (:py:class:`str`, callable)
        A list of doubles in which the first entry are names of each processing
        step in the pipeline and second entry must be default-constructible
        :py:class:`bob.bio.base.preprocessor.Preprocessor` or
        :py:class:`bob.bio.base.preprocessor.Extractor` in any order.  Each
        of these objects must be a python type, that can be instantiated and
        used through its ``__call__()`` interface to process a single entry of
        a sample.  For python types that you may want to plug-in, but do not
        offer a default constructor that you like, pass the result of
        :py:func:`functools.partial` instead.

    """

    def __init__(self, pipeline):
        super(SampleLoaderAnnotated, self).__init__(pipeline)


    def _handle_step(self, sset, func, checkpoint):
        """Handles a single step in the pipeline, with optional checkpointing

        Parameters
        ----------

        sset : SampleSet
            The original sample set to be processed (delayed or pre-loaded)

        func : callable
            The processing function to call for processing **each** sample in
            the set, if needs be

        checkpoint : str, None
            An optional string that may point to a directory that will be used
            for checkpointing the processing phase in question


        Returns
        -------

        r : SampleSet
            The prototype processed sample.  If no checkpointing required, this
            will be of type :py:class:`Sample`.  Otherwise, it will be a
            :py:class:`DelayedSample`

        """

        if checkpoint is not None:
            samples = []  # processed samples
            for s in sset.samples:
                # there can be a checkpoint for the data to be processed
                candidate = os.path.join(checkpoint, s.path + ".hdf5")
                if not os.path.exists(candidate):

                    # TODO: Fix this on bob.bio.base
                    try:
                        # preprocessing is required, and checkpointing, do it now
                        data = func(s.data, annotations=s.annotations)
                    except:                        
                        data = func(s.data)

                    # notice this can be called in parallel w/o failing
                    bob.io.base.create_directories_safe(
                        os.path.dirname(candidate)
                    )
                    # bob.bio.base standard interface for preprocessor
                    # has a read/write_data methods
                    writer = (
                        getattr(func, "write_data")
                        if hasattr(func, "write_data")
                        else getattr(func, "write_feature")
                    )
                    writer(data, candidate)

                # because we are checkpointing, we return a DelayedSample
                # instead of normal (preloaded) sample. This allows the next
                # phase to avoid loading it would it be unnecessary (e.g. next
                # phase is already check-pointed)
                reader = (
                    getattr(func, "read_data")
                    if hasattr(func, "read_data")
                    else getattr(func, "read_feature")
                )                
                if is_picklable(reader):
                    samples.append(
                        DelayedSample(
                            functools.partial(reader, candidate), parent=s
                        )
                    )
                else:                    
                    logger.warning(f"The method {reader} is not picklable. Shiping its unbounded method to `DelayedSample`.")
                    reader = reader.__func__ # The reader object might not be picklable

                    samples.append(
                        DelayedSample(
                            functools.partial(reader, None, candidate), parent=s
                        )
                    )

        else:
            # if checkpointing is not required, load the data and preprocess it
            # as we would normally do
            samples = [Sample(func(s.data), parent=s) for s in sset.samples]

        r = SampleSet(samples, parent=sset)
        return r
