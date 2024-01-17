import logging

try:
    import pytestqt
except:
    logging.warning("Could not import pytestqt, skipping GUI tests.")
    collect_ignore_glob = ["gui/*"]

from tests.fixtures.skeletons import *
from tests.fixtures.instances import *
from tests.fixtures.datasets import *
from tests.fixtures.videos import *
from tests.fixtures.models import *
from tests.fixtures.announcements import *
