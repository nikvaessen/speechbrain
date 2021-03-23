################################################################################
#
# Converts the unzipped wav/<SPEAKER_ID>/<YT_VIDEO_ID>/<UTT_NUM>.wav folder
# structure of voxceleb into a WebDataset format
#
# Author(s): Nik Vaessen
################################################################################

import json
import pathlib
import argparse
import random
import subprocess

from collections import defaultdict
from typing import Tuple

import torch
import torchaudio
import webdataset as wds

import yaspin

from util import remove_directory


################################################################################
# methods for writing the shards

ID_SEPARATOR = "&"


def load_audio(audio_file_path: pathlib.Path) -> torch.Tensor:
    t, sr = torchaudio.load(audio_file_path)

    if sr != 16000:
        raise ValueError("expected sampling rate of 16 kHz")

    return t


def write_shards(
    voxceleb_folder_path: pathlib.Path,
    shards_path: pathlib.Path,
    seed: int,
    delete_voxceleb_folder: bool,
    compress_in_place: bool,
    samples_per_shard: int,
):
    """

    Parameters
    ----------
    voxceleb_folder_path: folder where extracted voxceleb data is located
    shards_path: folder to write shards of data to
    seed: random seed used to initially shuffle data into shards
    delete_voxceleb_folder: boolean value determining whether the input folder
                            (voxceleb_folder_path) will be deleted after shards
                            have been written
    compress_in_place: boolean value determining whether the shards will be
                       compressed with the `gpig` utility.
    samples_per_shard: number of data samples to store in each shards.

    """
    # make sure output folder exist
    shards_path.mkdir(parents=True, exist_ok=True)

    # find all audio files
    audio_files = sorted([f for f in voxceleb_folder_path.rglob("*.wav")])

    # create tuples (unique_sample_id, speaker_id, path_to_audio_file)
    data_tuples = []

    # track statistics on data
    all_speaker_ids = set()
    youtube_id_per_speaker = defaultdict(list)
    sample_keys_per_speaker = defaultdict(list)

    for f in audio_files:
        # path should be
        # $voxceleb_folder_path/wav/speaker_id/youtube_id/utterance_id.wav
        speaker_id = f.parent.parent.name
        youtube_id = f.parent.name
        utterance_id = f.stem

        # create a unique key for this sample
        key = f"{speaker_id}{ID_SEPARATOR}{youtube_id}{ID_SEPARATOR}{utterance_id}"

        # store statistics
        all_speaker_ids.add(speaker_id)
        youtube_id_per_speaker[speaker_id].append(youtube_id)
        sample_keys_per_speaker[speaker_id].append(key)

        t = (key, speaker_id, f)
        data_tuples.append(t)

    # determine a specific speaker_id label for each speaker_id
    speaker_id_to_idx = {
        speaker_id: idx
        for idx, speaker_id in enumerate(sorted(all_speaker_ids))
    }

    # write a meta.json file which contains statistics on the data
    # which will be written to shards
    meta_dict = {
        "speaker_ids": list(all_speaker_ids),
        "speaker_id_to_idx": speaker_id_to_idx,
        "youtube_id_per_speaker": youtube_id_per_speaker,
        "sample_keys_per_speaker": sample_keys_per_speaker,
        "num_data_samples": len(data_tuples)
    }

    with (shards_path / "meta.json").open("w") as f:
        json.dump(meta_dict, f)

    # swap the speaker id for the speaker_id index in each tuple
    data_tuples = [
        (key, speaker_id_to_idx[speaker_id], f)
        for key, speaker_id, f in data_tuples
    ]

    # shuffle the tuples so that each shard has a large variety in speakers
    random.seed(seed)
    random.shuffle(data_tuples)

    # write shards
    all_keys = set()
    shards_path.mkdir(exist_ok=True, parents=True)
    pattern = str(shards_path / "shard") + "-%06d.tar"

    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for key, speaker_id_idx, f in data_tuples:
            # load the audio tensor
            tensor = load_audio(f)

            # verify key is unique
            assert key not in all_keys
            all_keys.add(key)

            # extract speaker_id, youtube_id and utterance_id from key
            speaker_id, youtube_id, utterance_id = key.split(ID_SEPARATOR)

            # create sample to write
            sample = {
                "__key__": key,
                "wav.pyd": tensor,
                "meta.json": {
                    "speaker_id": speaker_id,
                    "youtube_id": youtube_id,
                    "utterance_id": utterance_id,
                    "speaker_id_idx": speaker_id_idx,
                },
            }

            # write sample to sink
            sink.write(sample)

    # optionally delete the (unsharded) input data
    if delete_voxceleb_folder:
        with yaspin.yaspin(f"deleting {voxceleb_folder_path}"):
            remove_directory(voxceleb_folder_path)

    # optionally compress the .tar shards
    if compress_in_place:
        with yaspin.yaspin(
            text=f"compressing .tar files in {shards_path}"
        ) as spinner:
            for p in sorted(shards_path.glob("*.tar")):
                spinner.write(f"> compressing {p}")
                subprocess.call(
                    ["pigz", p.absolute()],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )


################################################################################
# define CLI

parser = argparse.ArgumentParser(
    description="Convert VoxCeleb to WebDataset shards"
)

parser.add_argument(
    "voxceleb_folder_path",
    type=pathlib.Path,
    help="directory containing the (unzipped) VoxCeleb dataset",
)
parser.add_argument(
    "shards_path", type=pathlib.Path, help="directory to write shards to"
)
parser.add_argument(
    "--seed",
    type=int,
    default=12345,
    help="random seed used for shuffling data before writing to shard",
)
parser.add_argument(
    "--delete_voxceleb_folder",
    action="store_true",
    default=False,
    help="delete the voxceleb data folder after shards have been written",
)
parser.add_argument(
    "--compress_in_place",
    action="store_true",
    default=False,
    help="compress each .tar to .tar.gz, deleting the .tar file in the process",
)
parser.add_argument(
    "--samples_per_shard",
    type=int,
    default=5000,
    help="the maximum amount of samples placed in each shard. The last shard "
    "will most likely contain fewer samples.",
)


################################################################################
# execute script

if __name__ == "__main__":
    args = parser.parse_args()

    write_shards(
        args.voxceleb_folder_path,
        args.shards_path,
        args.seed,
        args.delete_voxceleb_folder,
        args.compress_in_place,
        args.samples_per_shard,
    )
