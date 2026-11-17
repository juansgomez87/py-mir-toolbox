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
numpydoc_show_class_members = False
myst_enable_extensions = ["colon_fence", "dollarmath"]
nb_execution_mode = "cache"
nb_execution_timeout = 300
exclude_patterns = ["_build", "jupyter_execute"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "librosa": ("https://librosa.org/doc/latest/", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "py-mir-toolbox"
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
