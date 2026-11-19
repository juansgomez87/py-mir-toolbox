# API reference

How closely each function matches MIRtoolbox 1.8.2 is shown in the {doc}`parity`.

```{eval-rst}
.. currentmodule:: mirpy

Data
----

.. autosummary::
   :toctree: generated/

   Signal
   Audio
   get_data

Framing
-------

.. autosummary::
   :toctree: generated/

   frame
   FrameSpec

Signal primitives
-----------------

.. autosummary::
   :toctree: generated/

   audio
   spectrum
   Spectrum
   cepstrum
   Cepstrum
   autocor
   Autocor
   filterbank
   sum
   peaks
   Peaks

Dynamics
--------

.. autosummary::
   :toctree: generated/

   rms
   lowenergy

Timbre and spectral shape
-------------------------

.. autosummary::
   :toctree: generated/

   zerocross
   centroid
   spread
   skewness
   kurtosis
   flatness
   entropy
   rolloff
   brightness
   mfcc
   Mfcc
   flux

Statistics
----------

.. autosummary::
   :toctree: generated/

   mean
   std
   median
   stat
   histo
   Histogram

Batch processing
----------------

.. autosummary::
   :toctree: generated/

   batch
   list_audio

Export
------

.. autosummary::
   :toctree: generated/

   file_table
   frame_table
   to_dataframe
   to_csv
   to_arff

Citing
------

.. autosummary::
   :toctree: generated/

   cite
```
