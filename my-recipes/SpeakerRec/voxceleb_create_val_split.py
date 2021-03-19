################################################################################
#
# Create a train/validation/test split from the (unzipped)
# training and test dataset folder of voxceleb. Expected folder structure:
# wav/<SPEAKER_ID>/<YT_VIDEO_ID>/<UTT_NUM>.wav
#
# Author(s): Nik Vaessen
################################################################################

import pathlib
import argparse
import random
import shutil

from collections import defaultdict

from util import remove_directory

################################################################################
# method for creating split


def create_val_split(
    train_folder_path: pathlib.Path,
    validation_folder_path: pathlib.Path,
    seed: int,
    ratio: float,
    overwrite_existing_validation_folder: bool,
):
    """

    Parameters
    ----------
    train_folder_path:
    validation_folder_path:
    seed:
    ratio:
    overwrite_existing_validation_folder:

    """
    # set random seed
    random.seed(seed)

    # make sure validation folder exist
    validation_folder_path = validation_folder_path / "wav"

    if overwrite_existing_validation_folder:
        remove_directory(validation_folder_path)

    validation_folder_path.mkdir(parents=True, exist_ok=True)

    # for each speaker we randomly select youtube_ids until we have achieved
    # the desired amount of validation samples
    for speaker_folder in (train_folder_path / "wav").iterdir():
        if not speaker_folder.is_dir():
            continue

        # first determine all samples in each youtube_id folder
        files_dict = defaultdict(list)

        for youtube_id_folder in speaker_folder.iterdir():
            files_dict[youtube_id_folder] = [
                f for f in youtube_id_folder.glob("*.wav")
            ]

        # select which youtube_ids will be placed in validation folder
        total_samples = sum(len(samples) for samples in files_dict.values())
        potential_youtube_ids = sorted([yid for yid in files_dict.keys()])

        val_youtube_ids = []
        current_val_samples = 0

        while current_val_samples / total_samples <= ratio:
            # select 3 random ids
            candidates = []
            for _ in range(0, 3):
                if len(potential_youtube_ids) == 0:
                    break

                yid = potential_youtube_ids.pop(
                    random.randint(0, len(potential_youtube_ids) - 1)
                )
                candidates.append(yid)

            # take the smallest one to prevent exceeding the ratio by to much
            candidates = sorted(candidates, key=lambda c: len(files_dict[c]))
            smallest_yid = candidates.pop(0)
            val_youtube_ids.append(smallest_yid)
            current_val_samples += len(files_dict[smallest_yid])

            # put the other 2 back
            for yid in candidates:
                potential_youtube_ids.append(yid)

        # move the validation samples to the validation folder
        val_speaker_folder = validation_folder_path / speaker_folder.name
        val_speaker_folder.mkdir(exist_ok=False, parents=True)

        for val_youtube_id in val_youtube_ids:
            shutil.move(
                str(val_youtube_id),
                validation_folder_path
                / speaker_folder.name
                / val_youtube_id.name,
            )


################################################################################
# define CLI

parser = argparse.ArgumentParser(
    description="Split VoxCeleb into a training and validation folder"
)

parser.add_argument(
    "train_folder_path",
    type=pathlib.Path,
    help="directory containing the (unzipped) VoxCeleb training dataset",
)
parser.add_argument(
    "validation_folder_path",
    type=pathlib.Path,
    help="directory to move validation samples to",
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="random seed used for determining which samples are moved to the validation folder",
)
parser.add_argument(
    "--ratio",
    type=float,
    default=0.1,
    help="the percentage of the input data which should become validation."
    " This is will not be exact.",
)
parser.add_argument(
    "--overwrite_existing_validation_folder",
    default=False,
    action="store_true",
    help="delete an already existing folder in the given validation_folder_path",
)


################################################################################
# execute script

if __name__ == "__main__":
    args = parser.parse_args()

    create_val_split(
        args.train_folder_path,
        args.validation_folder_path,
        args.seed,
        args.ratio,
        args.overwrite_existing_validation_folder,
    )
