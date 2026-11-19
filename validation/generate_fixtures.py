"""Generate MIRtoolbox 1.8.2 reference outputs for the parity tests.

The fixtures are computed with pymirtoolbox, which runs a compiled MIRtoolbox
1.8.2 on the free MATLAB Runtime R2025b, so no MATLAB licence is needed.
Run it on Linux with the Runtime installed (see validation/README.md):

    python validation/generate_fixtures.py --audio-dir AUDIO_DIR --out-dir validation/fixtures

Each audio file and feature gives one ``<stem>__<feature>.npz`` file holding
the outputs of pymirtoolbox, together with the call parameters and the
software versions as metadata.
"""

import argparse
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

MIRTOOLBOX_VERSION = "1.8.2"
AUDIO_SUFFIXES = {".wav", ".flac", ".aiff", ".au"}

# Feature name -> (pymirtoolbox function, keyword arguments).
# Add entries as pymirtoolbox exposes more MIRtoolbox operators.
FEATURE_CALLS = {
    # "_runtime" keeps these apart from the MATLAB cases of validation/*_cases.tsv
    "brightness_frame_runtime": ("mirbrightness", {"Frame": True}),
    "mfcc_frame_runtime": ("mirmfcc", {"Frame": True}),
    "rolloff_frame_runtime": ("mirrolloff", {"Frame": True}),
    "zerocross_frame_runtime": ("mirzerocross", {"Frame": True}),
    "chromagram_frame_runtime": ("mirchromagram", {"Frame": True}),
    "keystrength_runtime": ("mirkeystrength", {}),
    "key_runtime": ("mirkey", {}),
    "mode_runtime": ("mirmode", {}),
    "tonalcentroid_frame_runtime": ("mirtonalcentroid", {"Frame": True}),
    "hcdf_runtime": ("mirhcdf", {}),
    "roughness_runtime": ("mirroughness", {}),
    "regularity_frame_runtime": ("mirregularity", {"Frame": True}),
    "pitch_frame_runtime": ("mirpitch", {"Frame": True}),
    "inharmonicity_frame_runtime": ("mirinharmonicity", {"Frame": True}),
    "events_runtime": ("mirevents", {}),
    "eventdensity_runtime": ("mireventdensity", {}),
    "tempo_runtime": ("mirtempo", {}),
    "pulseclarity_runtime": ("mirpulseclarity", {}),
    "fluctuation_runtime": ("mirfluctuation", {}),
    "beatspectrum_runtime": ("mirbeatspectrum", {}),
}


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def find_audio(audio_dir: Path) -> list[Path]:
    return sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)


def run_feature(extractor, func_name: str, kwargs: dict, audio: Path) -> dict[str, np.ndarray]:
    result = getattr(extractor, func_name)(audio_input=str(audio), **kwargs)
    return {key: np.asarray(value) for key, value in result.items()}


def save_fixture(out_path: Path, outputs: dict[str, np.ndarray], metadata: dict) -> None:
    np.savez_compressed(out_path, **outputs, metadata=json.dumps(metadata))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    from pymirtoolbox import feature_extractor

    args.out_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        "mirtoolbox": MIRTOOLBOX_VERSION,
        "pymirtoolbox": package_version("pymirtoolbox"),
        "numpy": np.__version__,
        "python": platform.python_version(),
    }
    for audio in find_audio(args.audio_dir):
        for feature, (func_name, kwargs) in FEATURE_CALLS.items():
            outputs = run_feature(feature_extractor, func_name, kwargs, audio)
            metadata = {"audio": audio.name, "function": func_name, "kwargs": kwargs, **versions}
            save_fixture(args.out_dir / f"{audio.stem}__{feature}.npz", outputs, metadata)
            print(f"{audio.name}: {feature} -> {sorted(outputs)}")


if __name__ == "__main__":
    main()
