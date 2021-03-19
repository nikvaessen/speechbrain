################################################################################
#
# Utility functions.
#
# Author(s): Nik Vaessen
################################################################################


import pathlib

################################################################################
# method to delete a folder using the Pathlib API.


def remove_directory(dir_path: pathlib.Path):
    for child in dir_path.glob("*"):
        if child.is_file():
            child.unlink()
        else:
            remove_directory(child)

    dir_path.rmdir()
