"""Citation information for py-mir-toolbox and the original MIRtoolbox."""

MIRTOOLBOX_REFERENCES = {
    "lartillot2007": """@inproceedings{lartillot2007,
  author    = {Lartillot, Olivier and Toiviainen, Petri},
  title     = {A Matlab Toolbox for Musical Feature Extraction from Audio},
  booktitle = {Proceedings of the 10th International Conference on Digital Audio Effects (DAFx-07)},
  address   = {Bordeaux, France},
  year      = {2007}
}""",
    "lartillot2008": """@incollection{lartillot2008,
  author    = {Lartillot, Olivier and Toiviainen, Petri and Eerola, Tuomas},
  title     = {A Matlab Toolbox for Music Information Retrieval},
  booktitle = {Data Analysis, Machine Learning and Applications},
  editor    = {Preisach, Christine and Burkhardt, Hans and Schmidt-Thieme, Lars and Decker, Reinhold},
  publisher = {Springer},
  year      = {2008}
}""",
}


def cite() -> str:
    """Return BibTeX entries for the original MIRtoolbox papers.

    The citation of py-mir-toolbox itself is in ``CITATION.cff`` in the
    source repository. Please cite both the original MIRtoolbox papers
    and py-mir-toolbox in academic work.

    Returns
    -------
    str
        BibTeX entries, separated by blank lines.

    Examples
    --------
    >>> import mirpy
    >>> print(mirpy.cite())  # doctest: +ELLIPSIS
    @inproceedings{lartillot2007,
    ...
    """
    return "\n\n".join(MIRTOOLBOX_REFERENCES.values())
