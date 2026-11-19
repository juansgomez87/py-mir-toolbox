"""Phase 2 parity: dynamics, timbre and spectral-shape features, statistics.

Two fixture sources:

- pymirtoolbox (``validation/generate_fixtures.py``): ``brightness_frame_runtime``,
  ``mfcc_frame_runtime``, ``rolloff_frame_runtime``, ``zerocross_frame_runtime``.
  Their arrays are named after the output (``brightness``, ``mfcc``, ...).
- every feature in ``validation/phase2_cases.tsv``, generated in MATLAB by
  ``validation/mirtoolbox_fixtures.m``. The mirpy calls below must mirror
  the MATLAB calls of that file; ``test_cases_match_the_matlab_case_list``
  checks that no case is missing on either side.
"""

import numpy as np
import pytest

import mirpy as mir
from parity.cases import (
    BINS_ALONG_FRAMES,
    STANDARD,
    cases_with_fixtures,
    check_signal,
    load_matlab_case,
)
from parity.harness import PHASE2_CASES, Stage, from_matlab_axes, read_cases

CALLS = read_cases(PHASE2_CASES)
STAT_PREFIX = "stat_"  # mirstat fields in MATLAB fixtures, e.g. stat_Mean


def audio(f):
    return mir.audio(f)


def framed_rms(f):
    return mir.rms(audio(f), frame=True)


def framed_mfcc(f):
    return mir.mfcc(audio(f), frame=True)


# feature -> (mirpy equivalent of the MATLAB call, tolerance stage)
CASES = {
    "rms": (lambda f: mir.rms(audio(f)), Stage.LINEAR),
    "rms_frame": (framed_rms, Stage.LINEAR),
    "rms_median": (lambda f: mir.rms(audio(f), frame=True, median=True), Stage.LINEAR),
    "lowenergy": (lambda f: mir.lowenergy(audio(f)), Stage.LINEAR),
    "lowenergy_threshold": (lambda f: mir.lowenergy(audio(f), threshold=0.5), Stage.LINEAR),
    "lowenergy_asr": (lambda f: mir.lowenergy(audio(f), asr=True), Stage.LINEAR),
    "zerocross": (lambda f: mir.zerocross(audio(f)), Stage.LINEAR),
    "zerocross_frame": (lambda f: mir.zerocross(audio(f), frame=True), Stage.LINEAR),
    "zerocross_sample_both": (
        lambda f: mir.zerocross(audio(f), frame=True, per="sample", direction="both"),
        Stage.LINEAR,
    ),
    "centroid": (lambda f: mir.centroid(audio(f)), Stage.LINEAR),
    "centroid_frame": (lambda f: mir.centroid(audio(f), frame=True), Stage.LINEAR),
    "centroid_nogate": (
        lambda f: mir.centroid(audio(f), frame=True, min_rms=0, max_entropy=1),
        Stage.LINEAR,
    ),
    "spread_frame": (lambda f: mir.spread(audio(f), frame=True), Stage.LINEAR),
    "skewness_frame": (lambda f: mir.skewness(audio(f), frame=True), Stage.LINEAR),
    "kurtosis_frame": (lambda f: mir.kurtosis(audio(f), frame=True), Stage.LINEAR),
    "flatness_frame": (lambda f: mir.flatness(audio(f), frame=True), Stage.LINEAR),
    "entropy_frame": (lambda f: mir.entropy(audio(f), frame=True), Stage.LINEAR),
    "entropy_center": (lambda f: mir.entropy(audio(f), frame=True, center=True), Stage.LINEAR),
    "rolloff_frame": (lambda f: mir.rolloff(audio(f), frame=True), Stage.LINEAR),
    "rolloff_95": (lambda f: mir.rolloff(audio(f), frame=True, threshold=0.95), Stage.LINEAR),
    "brightness": (lambda f: mir.brightness(audio(f)), Stage.LINEAR),
    "brightness_cutoff": (
        lambda f: mir.brightness(audio(f), frame=True, cutoff=3000),
        Stage.LINEAR,
    ),
    "mfcc": (lambda f: mir.mfcc(audio(f)), Stage.LINEAR),
    "mfcc_rank0": (lambda f: mir.mfcc(audio(f), frame=True, rank=range(13)), Stage.LINEAR),
    "mfcc_bands": (lambda f: mir.mfcc(audio(f), frame=True, bands=20), Stage.LINEAR),
    "mfcc_delta": (lambda f: mir.mfcc(audio(f), frame=True, delta=1), Stage.LINEAR),
    "mfcc_delta2": (
        lambda f: mir.mfcc(audio(f), frame=True, delta=2, radius=1),
        Stage.LINEAR,
    ),
    "flux": (lambda f: mir.flux(audio(f)), Stage.LINEAR),
    "flux_inc": (lambda f: mir.flux(audio(f), inc=True), Stage.LINEAR),
    "flux_city": (lambda f: mir.flux(audio(f), dist="cityblock"), Stage.LINEAR),
    "flux_complex": (lambda f: mir.flux(audio(f), complex=True), Stage.LINEAR),
    "flux_median": (
        lambda f: mir.flux(audio(f), median=(0.2, 1.0), halfwave=True),
        Stage.LINEAR,
    ),
    "flux_subband": (lambda f: mir.flux(audio(f), sub_band="manual"), Stage.FILTER),
    "flux_mfcc": (lambda f: mir.flux(framed_mfcc(f)), Stage.LINEAR),
    "histo": (lambda f: mir.histo(audio(f)), Stage.LINEAR),
    "histo_number": (lambda f: mir.histo(framed_rms(f), number=20), Stage.LINEAR),
    "histo_ampli": (lambda f: mir.histo(framed_rms(f), ampli=True), Stage.LINEAR),
    "mean_mfcc": (lambda f: mir.mean(framed_mfcc(f)), Stage.LINEAR),
    "std_mfcc": (lambda f: mir.std(framed_mfcc(f)), Stage.LINEAR),
    "median_mfcc": (lambda f: mir.median(framed_mfcc(f)), Stage.LINEAR),
}

# feature -> mirpy equivalent of the MATLAB mirstat call (returns a dict of fields)
STAT_CASES = {
    "stat_rms": lambda f: mir.stat(framed_rms(f)),
    "stat_centroid": lambda f: mir.stat(mir.centroid(audio(f), frame=True)),
    "stat_mfcc": lambda f: mir.stat(framed_mfcc(f)),
}

# pymirtoolbox fixture -> (mirpy equivalent, output array name)
RUNTIME_CASES = {
    "brightness_frame_runtime": (lambda f: mir.brightness(audio(f), frame=True), "brightness"),
    "mfcc_frame_runtime": (framed_mfcc, "mfcc"),
    "rolloff_frame_runtime": (lambda f: mir.rolloff(audio(f), frame=True), "rolloff"),
    "zerocross_frame_runtime": (lambda f: mir.zerocross(audio(f), frame=True), "zerocross"),
}


def test_cases_match_the_matlab_case_list():
    assert set(CASES) | set(STAT_CASES) == set(CALLS)


@pytest.mark.parametrize(("feature", "stem"), cases_with_fixtures(CASES))
def test_matlab_case(feature, stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    ref, path = load_matlab_case(feature, stem, CALLS, mirtoolbox_fixture, fixture_audio)
    build, stage = CASES[feature]
    layout = BINS_ALONG_FRAMES if feature.startswith("histo") else STANDARD
    check_signal(build(path), ref, stage, f"{stem}: {feature}", assert_parity, layout)


@pytest.mark.parametrize(("feature", "stem"), cases_with_fixtures(STAT_CASES))
def test_matlab_stat(feature, stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    ref, path = load_matlab_case(feature, stem, CALLS, mirtoolbox_fixture, fixture_audio)
    fields = STAT_CASES[feature](path)
    expected = {
        key.removeprefix(STAT_PREFIX): values
        for key, values in ref.arrays.items()
        if key.startswith(STAT_PREFIX)
    }
    assert expected, "fixture has no mirstat fields"
    for field, values in expected.items():
        assert field in fields, f"mirstat field {field} missing from mir.stat"
        label = f"{stem}: {feature} {field}"
        assert_parity(fields[field].data.ravel(), np.ravel(values), Stage.LINEAR, label)


@pytest.mark.parametrize(("feature", "stem"), cases_with_fixtures(RUNTIME_CASES))
def test_runtime_case(feature, stem, mirtoolbox_fixture, fixture_audio, assert_parity):
    ref = mirtoolbox_fixture(stem, feature)
    build, output = RUNTIME_CASES[feature]
    result = build(fixture_audio(ref))
    expected = from_matlab_axes(ref.arrays[output])
    assert_parity(result.data, expected, Stage.LINEAR, f"{stem}: {feature}")
