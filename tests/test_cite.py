import mirpy


def test_cite_includes_both_original_papers():
    bibtex = mirpy.cite()
    assert "@inproceedings{lartillot2007" in bibtex
    assert "@incollection{lartillot2008" in bibtex


def test_version_is_exposed():
    assert isinstance(mirpy.__version__, str)
