"""Sphinx configuration for the py-mir-toolbox documentation and course."""

import mirpy

project = "py-mir-toolbox"
author = "Juan Sebastián Gómez-Cañón and contributors"
copyright = "2026, " + author
release = mirpy.__version__
version = ".".join(release.split(".")[:2])

GITHUB_REPO = "juansgomez87/py-mir-toolbox"
SITE_URL = "https://ccrma.stanford.edu/~jsgomezc/py-mir-toolbox/"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
]

autosummary_generate = True


def _case_clash_filenames(names) -> dict[str, str]:
    """Separate stub files for names that differ only by case (``audio``/``Audio``).

    On case-insensitive filesystems (macOS) their stubs would overwrite each
    other; the lowercase function gets a ``-function`` suffix.
    """
    lowered = [name.lower() for name in names]
    return {
        f"mirpy.{name}": f"mirpy.{name}-function"
        for name in names
        if name.islower() and lowered.count(name) > 1
    }


autosummary_filename_map = _case_clash_filenames(mirpy.__all__)
numpydoc_show_class_members = False
myst_enable_extensions = ["colon_fence", "dollarmath"]
nb_execution_mode = "cache"
nb_execution_timeout = 300
nb_render_markdown_format = "myst"  # tables in generated Markdown (Parity report)
exclude_patterns = ["_build", "_redirects", "jupyter_execute", "parity_report.py"]
html_extra_path = ["_redirects"]  # old chapter URLs (B_basics.html, ...) forward to the new ones

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "librosa": ("https://librosa.org/doc/latest/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"py-mir-toolbox {version}"
html_baseurl = SITE_URL + "docs/"
html_theme_options = {
    "github_url": f"https://github.com/{GITHUB_REPO}",
    "external_links": [{"name": "Interactive site", "url": SITE_URL}],
    "use_edit_page_button": True,
    "navigation_with_keys": False,
}
html_context = {
    "github_user": GITHUB_REPO.split("/")[0],
    "github_repo": GITHUB_REPO.split("/")[1],
    "github_version": "main",
    "doc_path": "docs",
}
