import logging

try:
    import pytestqt
except:
    logging.warning('Could not import PySide2 or pytestqt, skipping GUI tests.')
    collect_ignore_glob = ["gui/*"]

