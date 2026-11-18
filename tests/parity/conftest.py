import pytest

from parity.harness import (
    compare,
    failure_message,
    find_audio,
    fixtures_dir,
    load_fixture,
    report_path,
    write_report,
)


@pytest.fixture(scope="session")
def parity_results():
    results = []
    yield results
    if results:
        write_report(results, report_path())


@pytest.fixture
def mirtoolbox_fixture():
    """Load ``<stem>__<feature>.npz``; skip the test when it is not available."""

    def load(stem: str, feature: str):
        path = fixtures_dir() / f"{stem}__{feature}.npz"
        if not path.exists():
            pytest.skip(f"MIRtoolbox fixture not found: {path}")
        return load_fixture(path)

    return load


@pytest.fixture
def record_parity(parity_results):
    """Compare and record for the parity report without asserting.

    For open questions (e.g. which of two conventions MIRtoolbox uses):
    both hypotheses are recorded and the report shows which one holds.
    """

    def record(actual, expected, stage, feature):
        result = compare(actual, expected, stage, feature)
        parity_results.append(result)
        return result

    return record


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
