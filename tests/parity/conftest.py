import pytest

from parity.harness import (
    compare,
    failure_message,
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
def assert_parity(parity_results):
    """Compare, record the result for the parity report, and assert."""

    def check(actual, expected, stage, feature):
        result = compare(actual, expected, stage, feature)
        parity_results.append(result)
        assert result.passed, failure_message(result)

    return check
