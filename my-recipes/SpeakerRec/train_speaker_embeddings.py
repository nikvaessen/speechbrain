#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict

import torch
import torchaudio
import speechbrain as sb
import webdataset as wds
from hyperpyyaml import load_hyperpyyaml

from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """
        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:
            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        spkid, _ = batch.spk_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            spkid = torch.cat([spkid] * self.n_augment, dim=0)

        loss = self.hparams.compute_cost(predictions, spkid, lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(uttid, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep_shards(hparams):
    train_shards_folder = Path(hparams["train_shards_folder"])
    val_shards_folder = Path(hparams["val_shards_folder"])

    # load the meta info json file
    with (train_shards_folder / "meta.json").open("r") as f:
        train_meta = json.load(f)
    with (val_shards_folder / "meta.json").open("r") as f:
        val_meta = json.load(f)

    # define the mapping functions in the data pipeline
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    def audio_pipeline(sample_dict: Dict):
        # unpack sample
        key = sample_dict["__key__"]
        meta = sample_dict["meta.json"]
        audio_tensor = sample_dict["wav.pyd"]

        # determine what part of audio sample to use
        audio_tensor = audio_tensor.squeeze()

        if hparams["random_chunk"]:
            start = random.randint(0, len(audio_tensor) - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = 0
            stop = len(audio_tensor)

        sig = audio_tensor[start:stop]

        # determine the speaker label of the sample
        spk_id_idx = torch.LongTensor([meta["speaker_id_idx"]])

        return {
            "sig": sig,
            "spk_id_encoded": spk_id_idx,
            "id": key,
        }

    # define the WebDatasets
    def find_urls(folder_path: str):
        return [str(f) for f in sorted(Path(folder_path).glob("shard-*.tar*"))]

    train_data = (
        wds.WebDataset(
            find_urls(train_shards_folder),
            length=train_meta["num_data_samples"],
        )
        .shuffle(1000)
        .decode("pil")
        .map(audio_pipeline)
    )
    print(f"training data consist of {train_meta['num_data_samples']} samples")

    valid_data = (
        wds.WebDataset(
            find_urls(val_shards_folder), length=val_meta["num_data_samples"]
        )
        .shuffle(1000)
        .decode("pil")
        .map(audio_pipeline)
    )
    print(f"validation data consist of {val_meta['num_data_samples']} samples")

    return train_data, valid_data


if __name__ == "__main__":
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_data_wds, valid_data_wds = dataio_prep_shards(hparams)

    # add collate_fn to dataloader options
    hparams['dataloader_options']['collate_fn'] = PaddedBatch

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    speaker_brain = SpeakerBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    speaker_brain.fit(
        speaker_brain.hparams.epoch_counter,
        train_data_wds,
        valid_data_wds,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )
