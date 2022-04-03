# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import shutil
import docs.utils

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "SLEAP"
author = "Talmo D. Pereira"
copyright = "2019–2022, Talmo Lab"

# The short X.Y version
version = "1.2.2"

# Get the sleap version
# with open("../sleap/version.py") as f:
#     version_file = f.read()
#     version = re.search("\d.+(?=['\"])", version_file).group(0)

# Release should be the full branch name
release = "v1.2.2"

html_title = f"SLEAP ({release})"
html_short_title = "SLEAP"
html_favicon = "_static/favicon.ico"
html_baseurl = "/develop/"

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    # https://myst-nb.readthedocs.io/en/latest/
    "myst_nb",
]


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "build", "_templates", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# Autosummary linkcode resolution
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
def linkcode_resolve(domain, info):
    """Resolve GitHub URLs for linkcode extension."""

    if domain != "py":
        return None

    if not info["module"]:
        return None

    try:
        filename = docs.utils.resolve(info["module"], info["fullname"])
        if filename is None:
            return None
        return f"https://github.com/murthylab/sleap/blob/{release}/{filename}"
    except:
        print(info)
        raise


autosummary_generate = True

# Enable automatic role inference as a Python object for autodoc.
# This automatically converts object references to their appropriate role,
# making it much easier (and more legible) to insert references in docstrings.
#   Ex: `MyClass` -> :class:`MyClass`
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-default_role
default_role = "py:obj"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = "furo"
html_theme = "sphinx_book_theme"

# Customization options.
# https://pradyunsg.me/furo/customisation/
html_theme_options = {
    # Set this to add a site-wide banner:
    # "announcement": "<em>Important</em> announcement!",
    # "light_logo": "logo.png",
    # "dark_logo": "logo.png",
    # https://sphinx-book-theme.readthedocs.io/en/stable/customize/index.html#theme-options
    "repository_url": "https://github.com/murthylab/sleap",
    "use_download_button": False,
    "use_fullscreen_button": False,
    "use_repository_button": True,
    "use_issues_button": True,
    "extra_navbar": "",
    # https://sphinx-book-theme.readthedocs.io/en/stable/launch.html
    # "launch_buttons": {
    #     "colab_url": "https://{your-colab-url}"
    # },
}

myst_number_code_blocks = ["python"]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    # "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    # "strikethrough",
    "substitution",
    "tasklist",
]

html_logo = "_static/logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# Napoleon settings
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
# admonition is a warning box, but rubric is just another heading without affecting
# sectioning of the doc (used when configs below are set to False).
# https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#directives
# https://pradyunsg.me/furo/reference/admonitions/
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
# Replace types with aliases with key: val pairs. Example:
#   napoleon_type_aliases = {"np.ndarray": "array"}
#   -> Replaces `np.ndarray` with `array`
napoleon_type_aliases = {}
napoleon_attr_annotations = True


# -- Extension configuration -------------------------------------------------
# We copy some the files in _static to an alternative location
# of docs/_static in order to have paths work for README.rst and index.rst

_docs_static_path = f"{os.environ['BUILDDIR']}/html/docs/_static"
if os.path.exists(_docs_static_path):
    shutil.rmtree(_docs_static_path)
shutil.copytree("_static", _docs_static_path)

# https://myst-nb.readthedocs.io/en/latest/use/config-reference.html
jupyter_execute_notebooks = "off"
