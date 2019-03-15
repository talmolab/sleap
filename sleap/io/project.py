"""
A LEAP project collects LEAP datasets (video data, labels, etc.), skeletons,
and models into single data structure. It is backed by an assortment of data
files stored in a single directory. It can represent several sessions of work
using the LEAP pipeline.
"""

class Project:
    def __init__(self, directory_path: str):
        self.directory_path = directory_path

        # Load any datasets present in the project directory into memory
