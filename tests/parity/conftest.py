import pytest

from parity.harness import (
    compare,
    failure_message,
    find_audio,
    fixtures_dir,
    load_fixture,
    report_path,
    run_metadata,
    write_report,
)


@pytest.fixture(scope="session")
def fixture_metadata():
    """Metadata of every fixture loaded in the session, for the report's run description."""
    return []


@pytest.fixture(scope="session")
def parity_results(fixture_metadata):
    results = []
    yield results
    if results:
        write_report(results, report_path(), run_metadata(fixture_metadata))


@pytest.fixture
def mirtoolbox_fixture(fixture_metadata):
    """Load ``<stem>__<feature>.npz``; skip the test when it is not available."""

    def load(stem: str, feature: str):
        path = fixtures_dir() / f"{stem}__{feature}.npz"
        if not path.exists():
            pytest.skip(f"MIRtoolbox fixture not found: {path}")
        fixture = load_fixture(path)
        fixture_metadata.append(fixture.metadata)
        return fixture

    return load


@pytest.fixture
def assert_parity(parity_results):
    """Compare, record the result for the parity report, and assert."""

    def check(actual, expected, stage, feature):
        result = compare(actual, expected, stage, feature)
        parity_results.append(result)
        assert result.passed, failure_message(result)

    return check


@pytest.fixture
def fixture_audio():
    """Return the path of a fixture's audio file; skip the test when it is not available."""

    def locate(fixture):
        name = fixture.metadata.get("audio")
        path = find_audio(name) if name else None
        if path is None:
            pytest.skip(f"Audio for the fixture not found: {name}")
        return path

    return locate
