#!/bin/sh

# Reset to the old variables when deactivating the environment
export LD_LIBRARY_PATH=$SLEAP_OLD_LD_LIBRARY_PATH
export XLA_FLAGS=$SLEAP_OLD_XLA_FLAGS
export NO_ALBUMENTATIONS_UPDATE=$SLEAP_OLD_NO_ALBUMENTATIONS_UPDATE