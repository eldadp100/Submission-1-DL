import os
import shutil

from torch.utils.data import Dataset
from torchaudio.datasets.utils import download_url, extract_archive
from typing import Tuple
import configs
import torchaudio
import torch
import numpy as np
import ntpath

down_sample_rate = 16000


class VCTK(Dataset):
    """ my new dataset to handle VCTK dataset inspired by previous implementations"""

    def __init__(
            self,
            root: str,
            download: bool = False,
            audio_ext=".wav",
            transforms=None
    ):
        url = "http://www.udialogue.org/download/VCTK-Corpus.tar.gz"
        folder_name = "VCTK-Corpus"
        archive = os.path.join(root, "{}.tar.gz".format(folder_name))

        self.memories = {}
        self.transforms = transforms
        self._path = os.path.join(root, folder_name)
        self._txt_dir = os.path.join(self._path, "txt")
        self._audio_dir = os.path.join(self._path, "wav48")
        self._audio_ext = audio_ext

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    download_url(url, root)
                extract_archive(archive, self._path)

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found. Please use `download=True` to download it.")

        # Extracting speaker IDs from the folder structure
        self._speaker_ids = sorted(os.listdir(self._txt_dir))
        self._samples = {}

        for speaker_id in self._speaker_ids:
            utterance_dir = os.path.join(self._txt_dir, speaker_id)
            speaker_samples = []
            for utterance_file in sorted(f for f in os.listdir(utterance_dir) if f.endswith(".txt")):
                utterance_id = os.path.splitext(utterance_file)[0]
                text_path = os.path.join(
                    self._txt_dir,
                    speaker_id,
                    f"{utterance_id}.txt",
                )
                audio_path = os.path.join(
                    self._audio_dir,
                    speaker_id,
                    f"{utterance_id}{self._audio_ext}",
                )

                speaker_samples.append({'txt': text_path, 'audio': audio_path, 'utterance_id': utterance_id})
            self._samples[speaker_id] = speaker_samples

        self.samples_index = [len(self._samples[self._speaker_ids[0]])]
        for i in range(1, len(self._speaker_ids)):
            self.samples_index.append(self.samples_index[-1] + len(self._samples[self._speaker_ids[i]]))

    def _load_text(self, file_path) -> str:
        with open(file_path) as file_path:
            return file_path.readlines()[0]

    def _load_audio(self, file_path) -> Tuple[torch.Tensor, int]:
        return torchaudio.load(file_path)

    def _load_sample(self, speaker_id: str, utterance_id: str):
        txt_path = os.path.join(
            self._txt_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}.txt"
        )
        audio_path = os.path.join(
            self._audio_dir,
            speaker_id,
            f"{speaker_id}_{utterance_id}_{self._audio_ext}"
        )

        utterance = self._load_text(txt_path)
        waveform, sample_rate = self._load_audio(audio_path)

        return waveform, sample_rate, utterance, speaker_id, utterance_id

    def __getitem__(self, n: int):
        if n not in self.memories:
            speaker_id, utterance_id = self.speakers[n]
            waveform, sample_rate, utterance, speaker_id, utterance_id = self._load_sample(speaker_id, utterance_id)
            if self.transforms is not None:
                for transform in self.transforms:
                    waveform = transform(waveform)
            self.memories[n] = (waveform, sample_rate, utterance, speaker_id, utterance_id)
        return self.memories[n]

    def __len__(self) -> int:
        return len(self._sample_ids)


def pad_audio(signal, dst_size):
    ret = torch.zeros(dst_size)
    ret[:signal.shape[0]] = signal[:dst_size]
    return ret


def load_audio(file_path, pad=50000):
    audio, sample_rate = torchaudio.load(file_path)
    padded_audio = pad_audio(audio.view(-1).contiguous(), pad)
    return padded_audio, sample_rate


def save_audio(signal, file_path):
    torchaudio.save(file_path, signal, down_sample_rate)  # TODO


def aggregate_signals(signals):
    ret_signal, _ = load_audio(signals[0]['audio'])

    for i in range(1, len(signals)):
        ret_signal += load_audio(signals[i]['audio'])[0]

    return ret_signal / len(signals)


def concat(signals):
    loaded_signals = []
    for s in signals:
        loaded_signals.append(load_audio(s['audio'])[0].view(-1))
    # what if I add empty speaker between and limit the receptive field? I would get new SSL!
    # I sometimes (need to make it 50% of the cases) choose same speaker and sometimes different
    empty_voice_in_between_size = 10000
    ret_size = sum([s.shape[0] for s in signals]) + empty_voice_in_between_size * (len(signals) - 1)
    ret_signal = torch.zeros(ret_size)
    c = 0
    for i in range(len(loaded_signals)):
        add_size = loaded_signals[i].shape[0]
        ret_signal[c:c + add_size] = loaded_signals[i]
        c += add_size + empty_voice_in_between_size

    return ret_signal
    #
    # ret_signal_size = 0
    # for s in signals:
    #     ret_signal_size += s['audio']
    #
    # for i in range(1, len(signals)):
    #     ret_signal += load_audio(signals[i]['audio'])[0]
    #
    # return ret_signal / len(signals)
    #

def generate_talks_without_replacement(voices_dataset: VCTK, C, talkers, N=10000,
                                       tmp_aggregation_path=os.path.join(configs.SP_data_path, 'tmp_aggregations')):
    talks = []
    talks_groups = torch.randint(0, len(talkers) - 1, (N, C))
    gi = 0
    for group in talks_groups:
        group_chosen_samples = []
        for i in range(len(group)):
            speaker_id = talkers[group[i]]
            choose_from = voices_dataset._samples[speaker_id]
            idx = np.random.randint(0, len(choose_from) - 1)
            group_chosen_samples.append(choose_from[idx])
        agg_sample = aggregate_signals(group_chosen_samples)
        agg_sample_path = os.path.join(tmp_aggregation_path, f'group_{gi}.wav')
        save_audio(agg_sample, agg_sample_path)
        talks.append((agg_sample_path, group_chosen_samples))
        gi += 1
    return talks


train_test_talkers_ratio = 0.7


def save_generated_dataset(SP_path, dataset, group_shift=0):
    i = group_shift
    for agg_file, group in dataset:
        talk_path = os.path.join(SP_path, f'talk_{i}')
        if os.path.exists(talk_path):
            shutil.rmtree(talk_path)
        os.mkdir(talk_path)
        os.rename(agg_file, os.path.join(talk_path, 'agg.wav'))
        for j, file_path in enumerate(group):
            name = file_path['utterance_id']
            shutil.copy(file_path['audio'], os.path.join(talk_path, f'record_{j}_{name}.wav'))
        i += 1


def generate_datasets(voices_dataset: VCTK, C):
    """
        We first divide the speakers. Then we generate disjoint train and test from those speakers.
        Then we sample from the other speakers.
        The 3 sets are disjoint.

        C is number of speakers.
        The data is given as (aggregated multi talkers signal, [the C input signals])
        TODO: aggregation method using GANs (like unpaired image to image but utilize the information we have)
    """
    all_talkers = voices_dataset._speaker_ids
    number_of_training_talkers = int(train_test_talkers_ratio * len(all_talkers))
    training_talkers = np.random.choice(all_talkers, number_of_training_talkers)
    testing_talkers = np.random.choice(all_talkers, len(all_talkers) - number_of_training_talkers)

    train_talkers_samples = generate_talks_without_replacement(voices_dataset, C, training_talkers)
    split_idx = int(train_test_talkers_ratio * len(train_talkers_samples))
    train_dataset = train_talkers_samples[:split_idx]
    test_known_dataset = train_talkers_samples[split_idx:]
    save_generated_dataset(os.path.join(configs.SP_data_path, 'train'), train_dataset)
    save_generated_dataset(os.path.join(configs.SP_data_path, 'test_known_speakers'), test_known_dataset,
                           group_shift=split_idx)

    test_unknown_dataset = generate_talks_without_replacement(voices_dataset, C, testing_talkers, N=1000)
    save_generated_dataset(os.path.join(configs.SP_data_path, 'test_unknown_speakers'), test_unknown_dataset)

    return train_dataset, test_known_dataset, test_unknown_dataset


dataset = VCTK(root=configs.vctk_data_path, download=False,
               transforms=[torchaudio.transforms.Resample(48000, down_sample_rate)])
generate_datasets(dataset, 2)
