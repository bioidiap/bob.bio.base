..  _bob.bio.base.struct_bio_rec_sys:

============================================
Structure of a Biometric Recognition System
============================================

This section will familiarize you with the structure of a typical biometric recognition system to help you understand and use the ``bob.bio`` framework to set up your own biometric recognition experiments.

"Biometric recognition" refers to the process of establishing a person's identity based on their biometric data.
A biometric recognition system can operate in one of two modes: *verification* or *identification*.  
A *verification* system establishes whether or not a person is who they say they are (i.e., the person claims an identity and the system tries to prove whether or not that claim is true).
On the other hand, an *identification* system attempts to establish a person's identity from scratch (i.e., the system tries to associate a person with an identity from a set of identities in the system's database).  When we are talking about neither verification nor identification in particular, the generic term *recognition* is used. 

A biometric recognition system has two stages:

1. **Enrollment:** A person's biometric data is enrolled to the system's biometric database.
2. **Recognition:** A person's newly acquired biometric data (which we call a *probe*) is compared to the enrolled biometric data (which we refer to as a *model*), and a match score is generated.  The match score tells us how similar the model and the probe are.  Based on match scores, we then decide whether or not the model and probe come from the same person (verification) or which gallery identity should to be assigned to the input biometric (identification).

Fig. 1 shows the enrollment and verification stages in a typical biometric *verification* system:

.. figure:: /img/bio_ver_sys.svg
   :align: center

   Enrollment and verification in a typical biometric verification system.

Fig. 2 shows the enrollment and identification stages in a typical biometric *identification* system:

.. figure:: /img/bio_ident_sys.svg
   :align: center

   Enrollment and identification in a typical biometric identification system.

In the figures above:
* The "Pre-processor" cleans up the raw biometric data to make recognition easier (e.g., crops the face image to get rid of the background).
* The "Feature Extractor" extracts the most important features for recognition, from the pre-processed biometric data.
* The "Model Database" stores each person's extracted feature set in the form of a representative model for that person in the system database, typically alongside the person's ID.
* The "Matcher" compares a new biometric feature set (probe) to one (for verification) or all (for identification) models in the database, and outputs a similarity score for each comparison.
* For *verification*, the "Decision Maker" decides whether or not the probe and the model from the database match, based on whether the similarity score is above or below a pre-defined match threshold.  For *identification*, the "Decision Maker" decides which model from the database best represents the identity of the probe, based on which model most closely matches the probe.


Biometric Recognition Experiments in the ``bob.bio`` Framework
---------------------------------------------------------------

The ``bob.bio`` framework has the capability to perform both *verification* and *identification* experiments, depending on the user's requirements.  To talk about the framework in generic terms, we will henceforth use the term *recognition*.

In general, the goal of a biometric recognition experiment is to quantify the recognition accuracy of a biometric recognition system, e.g., we wish to find out how good the system is at deciding whether or not two biometric samples come from the same person.

To conduct a biometric recognition experiment, we need biometric data.  So, we use a biometric database.  A biometric database generally consists of multiple samples of a particular biometric, from multiple people.  For example, a face database could contain 5 different images of a person's face, from 100 people.  The dataset is split up into samples used for enrollment, and samples used for probing.  We then simulate "genuine" recognition attempts by comparing each person's probe samples to their enrolled models.  We simulate "impostor" recognition attempts by comparing the same probe samples to models of different people.

In ``bob.bio``, biometric recognition experiments are split up into four main stages, similar to the stages in a typical verification or identification system as illustrated in Fig. 1 and Fig. 2, respectively:

1. Data preprocessing
2. Feature extraction
3. Matching
4. Decision making

Each of these stages is discussed below:

Data Preprocessing:
~~~~~~~~~~~~~~~~~~~

Biometric measurements are often noisy, containing redundant information that is not necessary (and can be misleading) for recognition.  For example, face images contain non-face background information, vein images can be unevenly illuminated, speech signals can be littered with background noise, etc.  The aim of the data preprocessing stage is to clean up the raw biometric data so that it is in the best possible state to make recognition easier.  For example, biometric data is cropped from the background, the images are photometrically enhanced, etc.

All the biometric samples in the input biometric database go through the preprocessing stage.  The results are stored in a directory entitled "preprocessed".  This process is illustrated in Fig. 3:

.. figure:: /img/preprocessor.svg
   :align: center

   Preprocessing stage in ``bob.bio``'s biometric recognition experiment framework.


Feature Extraction:
~~~~~~~~~~~~~~~~~~~

Although the preprocessing stage produces cleaner biometric data, the resulting data is usually very large and still contains much redundant information.  The aim of the feature extraction stage is to extract features that are necessary for recognizing a person.

All the biometric features stored in the "preprocessed" directory go through the feature extraction stage.  The results are stored in a directory entitled "extracted".  This process is illustrated in Fig. 4:

.. figure:: /img/extractor.svg
   :align: center

   Feature extraction stage in ``bob.bio``'s biometric recognition experiment framework.

Note that there is sometimes a feature extractor training stage prior to the feature extraction (to help the extractor learn which features to extract), but this is not always the case.


Matching:
~~~~~~~~~

The matching stage in ``bob.bio`` is referred to as the "Algorithm".  The Algorithm stage consists of three main parts:

(i) An optional "projection" stage after the feature extraction, as illustrated in Fig. 5.  This would be used if, for example, you wished to project your extracted biometric features into a lower-dimensional subspace prior to recognition.

.. figure:: /img/algorithm_projection.svg
   :align: center

   The projection part of the Algorithm stage in ``bob.bio``'s biometric recognition experiment framework.


(ii) Enrollment: The enrollment part of the Algorithm stage essentially works as follows.  One or more biometric samples per person is used to compute a representative "model" for that person, which essentially represents that person's identity.  To determine which of a person's biometric samples should be used to generate their model, we query our input biometric database.  The model is then calculated using the corresponding biometric features extracted in the Feature Extraction stage (or, optionally, our "projected" features).  Fig. 6 illustrates the enrollment part of the Algorithm module:

.. figure:: /img/algorithm_enrollment.svg
   :align: center

   The enrollment part of the Algorithm stage in ``bob.bio``'s biometric recognition experiment framework.

Note that there is sometimes a model enroller training stage prior to enrollment.  This is only necessary when you are trying to fit an existing model to a set of biometric features, e.g., fitting a UBM (Universal Background Model) to features extracted from a speech signal.  In other cases, the model is calculated from the features themselves, e.g., by averaging the feature vectors from multiple samples of the same biometric, in which case model enroller training is not necessary.


(iii) Scoring: The scoring part of the Algorithm stage essentially works as follows.  Each model is associated with a number of probes, so we first query the input biometric database to determine which biometric samples should be used as the probes for each model.  Every model is then compared to its associated probes (some of which come from the same person, and some of which come from different people), and a score is calculated for each comparison.  The score may be a distance, and it tells us how similar or dissimilar the model and probe biometrics are.  Ideally, if the model and probe come from the same biometric (e.g., two images of the same finger), they should be very similar, and if they come from different sources (e.g., two images of different fingers) then they should be very different.  Fig. 7 illustrates the scoring part of the Algorithm module:

.. figure:: /img/algorithm_scoring.svg
   :align: center

   The scoring part of the Algorithm stage in ``bob.bio``'s biometric recognition experiment framework.


Decision Making:
~~~~~~~~~~~~~~~~

The decision making stage in ``bob.bio`` is referred to as "Evaluation".  If we wish to perform *verification*, then the aim of this stage will be to make a decision as to whether each score calculated in the Matching stage indicates a "Match" or "No Match" between the particular model and probe biometrics.  If we wish to perform *identification*, then the aim of the evaluation stage will be to find the model which most closely matches the probe biometric.  

Once a decision has been made, we can quantify the overall performance of the particular biometric recognition system in terms of common metrics like the False Match Rate (FMR), False Non Match Rate (FNMR), and Equal Error Rate (EER) for verification, and Identification Rate (IR) for identification.  We can also view a visual representation of the performance in terms of plots like the Receiver Operating Characteristic (ROC) and Detection Error Trade-off (DET) for verification, Cumulative Match Characteristics (CMC) for closed-set identification, and Detection and Identification Rate (DIR) for open-set identification.  Fig. 7 illustrates the Evaluation stage:

.. figure:: /img/evaluation.svg
   :align: center

   Evaluation stage in ``bob.bio``'s biometric recognition experiment framework.


.. note::

   * The "Data Preprocessing" to "Matching" steps are carried out by ``bob.bio.base``s ``verify.py`` script.  The "Decision Making" step is carried out by ``bob.bio.base``'s ``evaluate.py`` script.  These scripts will be discussed in the next sections. 
   * The communication between any two steps in the recognition framework is file-based, usually using a binary HDF5_ interface, which is implemented, for example, in the :py:class:`bob.io.base.HDF5File` class.
   * The output of one step usually serves as the input of the subsequent step(s), as portrayed in Fig. 3 -- Fig. 7.
   * ``bob.bio`` ensures that the correct files are always forwarded to the subsequent steps.  For example, if you choose to implement a feature projection after the feature extraction stage, as illustrated in Fig. 5, ``bob.bio`` will make sure that the files in the "projected" directory are passed on as the input to the Enrollment stage; otherwise, the "extracted" directory will become the input to the Enrollment stage.

.. include:: links.rst