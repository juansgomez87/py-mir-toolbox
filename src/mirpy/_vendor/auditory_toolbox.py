# Vendored from python_auditory_toolbox 1.0.6 (python_auditory_toolbox/auditory_toolbox.py),
# https://github.com/MalcolmSlaney/python_auditory_toolbox, by Malcolm Slaney.
# Licensed under the Apache License, Version 2.0; see LICENSE-APACHE-2.0.txt in this folder.
#
# Modified for py-mir-toolbox: only ErbSpace, MakeErbFilters and ErbFilterBank are kept
# (unchanged); the matplotlib import and the other functions were removed. Vendored
# because the PyPI package requires jax and torch, which these functions do not use.
"""Gammatone (ERB) filterbank from Slaney's Auditory Toolbox, Python port."""

import numpy as np
from scipy import signal

from typing import List


def ErbSpace(low_freq: float = 100, high_freq: float = 44100/4,
             n: int = 100) -> np.ndarray:
  """This function computes an array of N frequencies uniformly spaced between
  high_freq and low_freq on an erb scale.  N is set to 100 if not specified.

  See also linspace, logspace, MakeErbCoeffs, MakeErbFilters.

  For a definition of erb, see Moore, B. C. J., and Glasberg, B. R. (1983).
  "Suggested formulae for calculating auditory-filter bandwidths and
  excitation patterns," J. Acoust. Soc. Am. 74, 750-753.

  Args:
    low_freq: The center frequency in Hz of the lowest channel
    high_freq: The upper limit in Hz of the channel bank.  The center frequency
      of the highest channel will be below this frequency.
    n: Number of channels

  Returns:
    An array of center frequencies, equally spaced on the ERB scale.
  """

  #  Change the following three parameters if you wish to use a different
  #  erb scale.  Must change in MakeerbCoeffs too.
  ear_q = 9.26449				# Glasberg and Moore Parameters
  min_bw = 24.7

  # All of the follow_freqing expressions are derived in Apple TR #35, "An
  # Efficient Implementation of the Patterson-Holdsworth Cochlear
  # Filter Bank."  See pages 33-34.
  cf_array = (-(ear_q*min_bw) +
             np.exp(np.arange(1, 1+n)*
                    (-np.log(high_freq + ear_q*min_bw) +
                     np.log(low_freq + ear_q*min_bw))/n) * (high_freq +
                                                             ear_q*min_bw))
  return cf_array


def MakeErbFilters(fs: float, num_channels: int,
                   low_freq: float = 20) -> List[np.ndarray]:
  """This function computes the filter coefficients for a bank of
  Gammatone filters.  These filters were defined by Patterson and
  Holdworth for simulating the cochlea.

  The result is returned as an array of filter coefficients.  Each row
  of the filter arrays contains the coefficients for four second order
  filters.  The transfer function for these four filters share the same
  denominator (poles) but have different numerators (zeros).  All of these
  coefficients are assembled into one vector that the ErbFilterBank
  can take apart to implement the filter.

  The filter bank contains "num_channels" channels that extend from
  half the sampling rate (fs) to "low_freq".  Alternatively, if the num_channels
  input argument is a vector, then the values of this vector are taken to
  be the center frequency of each desired filter.  (The low_freq argument is
  ignored in this case.)

  Note this implementation fixes a problem in the original code by
  computing four separate second order filters.  This avoids a big
  problem with round off errors in cases of very small cfs (100Hz) and
  large sample rates (44kHz).  The problem is caused by roundoff error
  when a number of poles are combined, all very close to the unit
  circle.  Small errors in the eigth order coefficient, are multiplied
  when the eigth root is taken to give the pole location.  These small
  errors lead to poles outside the unit circle and instability.  Thanks
  to Julius Smith for leading me to the proper explanation.

  Execute the following code to evaluate the frequency
  response of a 10 channel filterbank.
    n = 512
    fs = 16000
    fcoefs = pat.MakeErbFilters(16000,10,100)
    y = pat.ErbFilterBank(np.array([1.0] + [0] * (n-1), dtype=float), fcoefs)
    resp = 20*np.log10(np.abs(np.fft.fft(y, axis=1))).T
    freq_scale = np.expand_dims(np.linspace(0, 16000, 512), 1)
    plt.semilogx(freq_scale[:n//2, :], resp[:n//2, :])
    plt.axis((100, fs/2, -60, 0))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Filter Response (dB)');

  Args:
    fs: Sampling rate (in Hz) of the filterbank (needed to determine CFs).
    num_channel: How many channels in the filterbank.
    low_freq: The lowest center frequency of the filterbank.

  Returns: 
    A list of 10 num_channel-D arrays containing the filter parameters.
  """

  t = 1/fs
  if isinstance(num_channels, int):
    cf = ErbSpace(low_freq, fs/2, num_channels)
  else:
    cf = num_channels

  # So equations below match the original Matlab syntax
  pi = np.pi
  abs = np.abs  # pylint: disable=redefined-builtin
  sqrt = np.sqrt
  sin = np.sin
  cos = np.cos
  exp = np.exp
  i = np.array([1j], dtype=np.csingle)

  # Change the follow_freqing three parameters if you wish to use a different
  # erb scale.  Must change in ErbSpace too.
  ear_q = 9.26449				#  Glasberg and Moore Parameters
  min_bw = 24.7
  order = 1

  erb = ((cf/ear_q)**order + min_bw**order)**(1/order)
  b=1.019*2*pi*erb

  a0 = t   # Feedback coefficients (poles)
  a2 = 0
  b0 = 1   # Feedforward coefficients (zeros)
  b1 = -2*cos(2*cf*pi*t)/exp(b*t)
  b2 = exp(-2*b*t)

  a11 = -(2*t*cos(2*cf*pi*t)/exp(b*t) + 2*sqrt(3+2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2
  a12 = -(2*t*cos(2*cf*pi*t)/exp(b*t) - 2*sqrt(3+2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2
  a13 = -(2*t*cos(2*cf*pi*t)/exp(b*t) + 2*sqrt(3-2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2
  a14 = -(2*t*cos(2*cf*pi*t)/exp(b*t) - 2*sqrt(3-2**1.5)*t*sin(2*cf*pi*t)/
      exp(b*t))/2

  gain = abs((-2*exp(4*i*cf*pi*t)*t +
                  2*exp(-(b*t) + 2*i*cf*pi*t)*t*
                          (cos(2*cf*pi*t) - sqrt(3 - 2**(3/2))*
                            sin(2*cf*pi*t))) *
            (-2*exp(4*i*cf*pi*t)*t +
              2*exp(-(b*t) + 2*i*cf*pi*t)*t*
                (cos(2*cf*pi*t) + sqrt(3 - 2**(3/2)) *
                sin(2*cf*pi*t)))*
            (-2*exp(4*i*cf*pi*t)*t +
              2*exp(-(b*t) + 2*i*cf*pi*t)*t*
                (cos(2*cf*pi*t) -
                sqrt(3 + 2**(3/2))*sin(2*cf*pi*t))) *
            (-2*exp(4*i*cf*pi*t)*t + 2*exp(-(b*t) + 2*i*cf*pi*t)*t*
            (cos(2*cf*pi*t) + sqrt(3 + 2**(3/2))*sin(2*cf*pi*t))) /
            (-2 / exp(2*b*t) - 2*exp(4*i*cf*pi*t) +
            2*(1 + exp(4*i*cf*pi*t))/exp(b*t))**4)

  allfilts = np.ones(len(cf))
  fcoefs = [a0*allfilts, a11, a12, a13, a14, a2*allfilts,
            b0*allfilts, b1, b2, gain]
  return fcoefs


def ErbFilterBank(x: np.ndarray, fcoefs: List[np.ndarray]) -> np.ndarray:
  """Filter an input signal with a filterbank, producing one output vector
  per channel.
  
  Args:
    x: The input signal, one-dimensional
    fcoefs: A list of 10 num-channel-dimensional arrays that describe the 
      filterbank.
  
  Returns:
    num-channel outputs in a num_channel x time array.
  """
  [a0, a11, a12, a13, a14, a2, b0, b1, b2, gain] = fcoefs
  n_chan = a0.shape[0]
  assert n_chan == a11.shape[0]
  assert n_chan == a12.shape[0]
  assert n_chan == a13.shape[0]
  assert n_chan == a14.shape[0]
  assert n_chan == b0.shape[0]
  assert n_chan == b1.shape[0]
  assert n_chan == b2.shape[0]
  assert n_chan == gain.shape[0]

  sos = np.stack((np.stack([a0/gain,  a0,   a0, a0], axis=1),
                  np.stack([a11/gain, a12, a13, a14], axis=1),
                  np.stack([a2/gain,  a2,   a2, a2], axis=1),
                  np.stack([b0, b0, b0, b0], axis=1),
                  np.stack([b1, b1, b1, b1], axis=1),
                  np.stack([b2, b2, b2, b2], axis=1)),
                axis=2)

  all_y = None
  for c in range(n_chan):
    y = signal.sosfilt(sos[c, :, :], x)
    if all_y is None:
      all_y = np.zeros((n_chan, len(y)), dtype=x.dtype)
    all_y[c, :] = y
  return all_y


