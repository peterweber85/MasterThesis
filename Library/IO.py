
import os
from fnmatch import fnmatch


def list_all_files_in_subdirectories(root_folder, pattern = "*.png"):

    filenames = []
    for path, subdirs, files in os.walk(root_folder):
        for name in files:
            if fnmatch(name, pattern):
                filenames.append(name)
    return filenames