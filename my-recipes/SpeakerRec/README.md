# Speaker recognition experiments with VoxCeleb.

This folder contains scripts for running speaker identification and verification experiments with the VoxCeleb dataset (http://www.robots.ox.ac.uk/~vgg/data/voxceleb/).

# Downloading and preparing data

In the following steps we assume a root data folder `~/data/`, which you can change to your liking. If you're using voxceleb2 the files need to be converted; this requires FFMPEG to be installed on your system.
Check for a correct installation of FFMPEG on your system by using `ffmpeg -version`.

1. Create a folder `~/data/voxceleb1` and `~/data/voxceleb2`. 
2. Following the instructions at http://www.robots.ox.ac.uk/~vgg/data/voxceleb/ to download voxceleb1 and/or voxceleb2. 
   You should end up with these 4 files:
    * `vox1_dev_wav.zip`
    * `vox1_test_wav.zip`
    * `vox2_dev_aac.zip`
    * `vox2_test_aac.zip`

Follow these instructions if you're interested in using voxceleb1:

3. Create a folder `~/data/voxceleb1/train` and `~/data/voxceleb1/test`
4. Move `vox1_dev_wav.zip` to `~/data/voxceleb1/train` and extract the archive. Optionally delete the archive after extraction.
5. Move `vox1_test_wav.zip` to `~/data/voxceleb1/test` and extract the archive. Optionally delete the archive after extraction.

Follow these instructions if you're interested in using voxceleb2:

6. Create a folder `~/data/voxceleb2/train` and `~/data/voxceleb2/test`
7. Move `vox2_dev_wav.zip` to `~/data/voxceleb2/train` and extract the archive. Optionally delete the archive after extraction.
8. Move `vox2_test_wav.zip` to `~/data/voxceleb2/test` and extract the archive. Optionally delete the archive after extraction.
9. Call `python voxceleb2_convert_to_wav.py ~/data/voxceleb2 --num_workers <NUM_CPUS_AVAILABLE>` to convert the files of voxceleb2 to the required format. This can take a few hours!

Now we need to create a train/val split and shard the data. It is important to decide which test split you are going to use right now.
We will assume test split `veri_test2.txt`, which only uses the data from `~/data/voxceleb1/test` as test data. 
Note that `list_test_hard2.txt` and `list_test_all2.txt` will use data from `~/data/voxceleb1/train` as well. 
In this case alter step 10 and 11 accordingly.

10. Create a folder `~/data/voxceleb/train/unzipped` and `mv ~/data/voxceleb1/train ~/data/voxceleb2/train ~/data/voxceleb/train/unzipped` 
11. Create a folder `~/data/voxceleb/test/unzipped` and `mv ~/data/voxceleb1/test ~/data/voxceleb/test/unzipped` 
12. Call `python voxceleb_create_val_split.py ~/data/voxceleb/train/unzipped ~/data/voxceleb/val/unzipped --ratio 0.1` to move 10% of your training data to a newly created validation folder.
13. Call `python voxceleb_create_webdatasets.py ~/data/voxceleb/train/unzipped ~/data/voxceleb/train` to shard your training data.
14. Call `python voxceleb_create_webdatasets.py ~/data/voxceleb/val/unzipped ~/data/voxceleb/val` to shard your validation data.
15. Call `python voxceleb_create_webdatasets.py ~/data/voxceleb/test/unzipped ~/data/voxceleb/test` to shard your test data.
16. Optionally delete the `~/data/voxceleb/*/unzipped` folders as well as `~/data/voxceleb1` and `~/data/voxceleb2`.

Note that you can use `python voxceleb_create_webdatasets.py <in_folder> <out_folder> --compress_in_place` to use compressed shards, which saves around 50% of storage place but slows down training.

# Training Xvectors

Run the following command to train xvectors:

`python train_speaker_embeddings.py hyperparams/train_x_vectors.yaml`

You can use the same script for voxceleb1, voxceleb2, and voxceleb1+2. Just change the datafolder and the corresponding number of speakers (1211 vox1, 5994 vox2, 7205 vox1+2).
For voxceleb1 + voxceleb2, see preparation instructions below).

It is possible to train embeddings with more augmentation with the following command:

`python train_speaker_embeddings.py hyperparams/train_ecapa_tdnn_big.yaml`

In this case, we concatenate waveform dropout, speed change, reverberation, noise, and noise+rev. The batch is 6 times larger than the original one. This normally leads to
a performance improvement at the cost of longer training time.

The system trains a TDNN for speaker embeddings coupled with a speaker-id classifier. The speaker-id accuracy should be around 97-98% for both voxceleb1 and voceleb2.

# Speaker verification with PLDA
After training the speaker embeddings, it is possible to perform speaker verification using PLDA.  You can run it with the following command:

`python speaker_verification_plda.py hyperparams/verification_plda_xvector.yaml`

If you didn't train the speaker embedding before, we automatically download the xvector model from the web.
This system achieves an EER = 3.2% on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)


# Speaker verification using ECAPA-TDNN embeddings
Run the following command to train speaker embeddings using [ECAPA-TDNN](https://arxiv.org/abs/2005.07143):

`python train_speaker_embeddings.py hyperparams/train_ecapa_tdnn.yaml`


The speaker-id accuracy should be around 98-99% for both voxceleb1 and voceleb2.

After training the speaker embeddings, it is possible to perform speaker verification using cosine similarity.  You can run it with the following command:

`python speaker_verification_cosine.py hyperparams/verification_ecapa_tdnn.yaml`

This system achieves an EER = 0.69 % on voxceleb1 + voxceleb2.
These results are all obtained with the official verification split of voxceleb1 (veri\_split.txt)

# VoxCeleb2 preparation
Voxceleb2 audio files are released in ma4 format. All the files must be converted in wav files before
feeding them is SpeechBrain. Please, follow these steps to prepare the dataset correctly:

1. Download both Voxceleb1 and Voxceleb2.
You can find download instructions here: http://www.robots.ox.ac.uk/~vgg/data/voxceleb/
Note that for the speaker verification experiments with Voxceleb2 the official split of voxceleb1 is used to compute EER.

2. Convert .ma4 to wav
Voxceleb2 stores files with the ma4 audio format. To use them within SpeechBrain you have to convert all the ma4 files into wav files.
You can do the conversion using ffmpeg(see for instance conversion scripts in https://gitmemory.com/issue/pytorch/audio/104/493137979 or https://gist.github.com/seungwonpark/4f273739beef2691cd53b5c39629d830). This operation might take several hours and should be only once.

2. Put all the wav files in a folder called wav. You should have something like `voxceleb2/wav/id*/*.wav` (e.g, `voxceleb2/wav/id00012/21Uxsk56VDQ/00001.wav`)

3. copy the `voxceleb1/vox1_test_wav.zip` file into the voxceleb2 folder.

4. Unpack voxceleb1 test files(verification split).

Go to the voxceleb2 folder and run `unzip vox1_test_wav.zip`.

5. Copy the verification split(`voxceleb1/ meta/veri_test.txt`) into voxceleb2(`voxceleb2/meta/ veri_test.txt`)

6. Now everything is ready and you can run voxceleb2 experiments:
- training embeddings:

`python train_speaker_embeddings.py hyperparams/train_xvector_voxceleb2.yaml`

Note: To prepare the voxceleb1 + voxceleb2 dataset you have to copy and unpack vox1_dev_wav.zip for the voxceleb1 dataset.

# Performance summary

[Speaker Verification Results with Voxceleb 1 + Voxceleb2]
| System          | Dataset    | EER  | Link |
|-----------------|------------|------| -----|
| Xvector + PLDA  | VoxCeleb 1,2 | 3.2% | - |
| ECAPA-TDNN      | Voxceleb 1,2 | 0.69% | - |


