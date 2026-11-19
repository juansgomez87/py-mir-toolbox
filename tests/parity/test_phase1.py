"""Phase 1 parity: audio options, framing, spectrum, autocor, cepstrum, filterbank, sum, peaks.

Two fixture sources:

- ``mfcc_frame_runtime`` (pymirtoolbox): mirmfcc's second output is, per the
  MIRtoolbox manual, "the output of mirspectrum, in mel-band and
  log-scale" (40 bands, 50 ms frames, half overlap).
- every feature in ``validation/phase1_cases.tsv``, generated in MATLAB by
  ``validation/mirtoolbox_fixtures.m``. The mirpy calls below must mirror the
  MATLAB calls of that file; ``test_cases_match_the_matlab_case_list``
  checks that no case is missing on either side.
"""

import numpy as np
import pytest

import mirpy as mir
from mirpy import FrameSpec
from parity.cases import (
    BANDS_AS_CHANNELS,
    STANDARD,
    cases_with_fixtures,
    check_signal,
    load_matlab_case,
)
from parity.harness import Stage, fixture_stems, from_matlab_axes, matlab_peaks, read_cases

CALLS = read_cases()
BAND_SPECTRA = {"spectrum_mel", "spectrum_bark", "spectrum_bark_mask"}
SCHEIRER_CUTOFFS = [-np.inf, 200, 400, 800, 1600, 3200, np.inf]


def spectrum_frames(f, **options):
    return mir.spectrum(mir.audio(f), frame=True, **options)


def autocor_frames(f, **options):
    return mir.autocor(mir.audio(f), frame=True, **options)


# feature -> (mirpy equivalent of the MATLAB call, tolerance stage)
CASES = {
    "audio": (lambda f: mir.audio(f), Stage.LINEAR),
    "audio_mono0": (lambda f: mir.audio(f, mono=False), Stage.LINEAR),
    "audio_center": (lambda f: mir.audio(f, center=True), Stage.LINEAR),
    "audio_sampling": (lambda f: mir.audio(f, sampling=11025), Stage.FILTER),
    "audio_extract": (lambda f: mir.audio(f, extract=(0.5, 1.25)), Stage.LINEAR),
    "audio_extract_sp": (
        lambda f: mir.audio(f, extract=(1000, 5000), extract_unit="sp"),
        Stage.LINEAR,
    ),
    "audio_trim": (lambda f: mir.audio(f, trim=True), Stage.LINEAR),
    "audio_trimstart": (lambda f: mir.audio(f, trim="start"), Stage.LINEAR),
    "audio_trimend": (lambda f: mir.audio(f, trim="end"), Stage.LINEAR),
    "audio_fwr": (lambda f: mir.audio(f, fwr=True), Stage.LINEAR),
    "frame": (lambda f: mir.frame(mir.audio(f)), Stage.LINEAR),
    "frame_samples": (
        lambda f: mir.frame(mir.audio(f), FrameSpec(1024, 0.25, "sp", "/1")),
        Stage.LINEAR,
    ),
    "spectrum": (lambda f: mir.spectrum(mir.audio(f)), Stage.LINEAR),
    "spectrum_frame": (lambda f: spectrum_frames(f), Stage.LINEAR),
    "spectrum_hanning": (lambda f: spectrum_frames(f, window="hanning"), Stage.LINEAR),
    "spectrum_minmax": (
        lambda f: spectrum_frames(f, min_freq=100, max_freq=5000),
        Stage.LINEAR,
    ),
    "spectrum_res": (lambda f: spectrum_frames(f, res=10), Stage.LINEAR),
    "spectrum_normal": (lambda f: spectrum_frames(f, normal=True), Stage.LINEAR),
    "spectrum_power": (lambda f: spectrum_frames(f, power=True), Stage.LINEAR),
    "spectrum_log": (lambda f: spectrum_frames(f, log=True), Stage.LINEAR),
    "spectrum_db": (lambda f: spectrum_frames(f, db=True), Stage.LINEAR),
    "spectrum_db40": (lambda f: spectrum_frames(f, db=40), Stage.LINEAR),
    "spectrum_mel": (lambda f: spectrum_frames(f, band="mel"), Stage.LINEAR),
    "spectrum_bark": (lambda f: spectrum_frames(f, band="bark"), Stage.LINEAR),
    "spectrum_bark_mask": (lambda f: spectrum_frames(f, band="bark", mask=True), Stage.LINEAR),
    "spectrum_terhardt": (lambda f: spectrum_frames(f, terhardt=True), Stage.LINEAR),
    "spectrum_smooth": (lambda f: spectrum_frames(f, smooth=5), Stage.LINEAR),
    "spectrum_gauss": (lambda f: spectrum_frames(f, gauss=3), Stage.LINEAR),
    "autocor": (lambda f: mir.autocor(mir.audio(f)), Stage.LINEAR),
    "autocor_frame": (lambda f: autocor_frames(f), Stage.LINEAR),
    "autocor_minmax": (
        lambda f: autocor_frames(f, min_lag=0.0012, max_lag=0.01),
        Stage.LINEAR,
    ),
    "autocor_hz": (lambda f: autocor_frames(f, min_freq=100, max_freq=1000), Stage.LINEAR),
    "autocor_nowindow": (lambda f: autocor_frames(f, normal_window=False), Stage.LINEAR),
    "autocor_biased": (lambda f: autocor_frames(f, scale="biased"), Stage.LINEAR),
    "autocor_compres": (lambda f: autocor_frames(f, generalized=0.67), Stage.LINEAR),
    "autocor_halfwave": (lambda f: autocor_frames(f, halfwave=True), Stage.LINEAR),
    "autocor_freq": (lambda f: autocor_frames(f, freq=True), Stage.LINEAR),
    "autocor_spectrum": (lambda f: mir.autocor(spectrum_frames(f)), Stage.LINEAR),
    "cepstrum_frame": (lambda f: mir.cepstrum(mir.audio(f), frame=True), Stage.LINEAR),
    "cepstrum_freq": (lambda f: mir.cepstrum(mir.audio(f), frame=True, freq=True), Stage.LINEAR),
    "filterbank": (lambda f: mir.filterbank(mir.audio(f)), Stage.FILTER),
    "filterbank_nb20": (lambda f: mir.filterbank(mir.audio(f), n_channels=20), Stage.FILTER),
    "filterbank_2channels": (lambda f: mir.filterbank(mir.audio(f), "2channels"), Stage.FILTER),
    "filterbank_manual": (
        lambda f: mir.filterbank(mir.audio(f), cutoffs=SCHEIRER_CUTOFFS),
        Stage.FILTER,
    ),
    "filterbank_scheirer": (
        lambda f: mir.filterbank(mir.audio(f), preset="scheirer"),
        Stage.FILTER,
    ),
    "filterbank_klapuri": (lambda f: mir.filterbank(mir.audio(f), preset="klapuri"), Stage.FILTER),
    "sum_filterbank": (lambda f: mir.sum(mir.filterbank(mir.audio(f))), Stage.FILTER),
    "sum_mean": (lambda f: mir.sum(mir.filterbank(mir.audio(f)), mean=True), Stage.FILTER),
    "sum_adjacent": (
        lambda f: mir.sum(mir.filterbank(mir.audio(f)), adjacent=3),
        Stage.FILTER,
    ),
}

# feature -> mirpy equivalent of the MATLAB mirpeaks call
PEAK_CASES = {
    "peaks_spectrum": lambda f: mir.peaks(spectrum_frames(f)),
    "peaks_spectrum_total": lambda f: mir.peaks(spectrum_frames(f), total=5),
    "peaks_spectrum_abscissa": lambda f: mir.peaks(spectrum_frames(f), order="abscissa"),
    "peaks_spectrum_contrast": lambda f: mir.peaks(spectrum_frames(f), contrast=0.3),
    "peaks_spectrum_threshold": lambda f: mir.peaks(spectrum_frames(f), threshold=0.5),
    "peaks_spectrum_valleys": lambda f: mir.peaks(spectrum_frames(f), valleys=True),
    "peaks_spectrum_noends": lambda f: mir.peaks(spectrum_frames(f), no_begin=True, no_end=True),
    "peaks_spectrum_reso": lambda f: mir.peaks(spectrum_frames(f), reso=100),
    "peaks_spectrum_local": lambda f: mir.peaks(spectrum_frames(f), normalize="local"),
    "peaks_autocor": lambda f: mir.peaks(autocor_frames(f)),
}


def test_cases_match_the_matlab_case_list():
    assert set(CASES) | set(PEAK_CASES) == set(CALLS)


@pytest.mark.parametrize(("feature", "stem"), cases_with_fixtures(CASES))
def test_matlab_case(feature, stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    ref, path = load_matlab_case(feature, stem, CALLS, mirtoolbox_fixture, fixture_audio)
    build, stage = CASES[feature]
    layout = BANDS_AS_CHANNELS if feature in BAND_SPECTRA else STANDARD
    check_signal(build(path), ref, stage, f"{stem}: {feature}", assert_parity, layout)


@pytest.mark.parametrize(("feature", "stem"), cases_with_fixtures(PEAK_CASES))
def test_matlab_peaks(feature, stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    ref, path = load_matlab_case(feature, stem, CALLS, mirtoolbox_fixture, fixture_audio)
    found = PEAK_CASES[feature](path).peaks
    expected = matlab_peaks(ref)
    label = f"{stem}: {feature}"
    assert_parity(found.channel, expected["channel"], Stage.EXACT, f"{label} channel")
    assert_parity(found.frame, expected["frame"], Stage.EXACT, f"{label} frame")
    assert_parity(found.index, expected["index"], Stage.DISCRETE, f"{label} index")
    assert_parity(found.pos, expected["precise_pos"], Stage.LINEAR, f"{label} position")
    # MIRtoolbox reports peak heights in signal units, not normalised (fixtures, 2026-11-19)
    assert_parity(found.val_original, expected["precise_val"], Stage.LINEAR, f"{label} val")


@pytest.mark.parametrize("stem", fixture_stems("mfcc_frame_runtime") or [""])
def test_mel_log_spectrum(stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    if not stem:
        pytest.skip("no MIRtoolbox fixture available")
    ref = mirtoolbox_fixture(stem, "mfcc_frame_runtime")
    x = mir.audio(fixture_audio(ref))
    mel = mir.spectrum(x, frame=True, band="mel", log=True)
    expected = from_matlab_axes(ref.arrays["melspec"])
    assert_parity(mel.data, expected, Stage.LINEAR, f"{stem}: mfcc_frame_runtime mel log spectrum")
