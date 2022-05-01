import random
import numpy as np
import torch
import torch.utils.data

import layers
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, dataset, hparams):
        self.dataset = dataset
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.stft = layers.TacotronSTFT(hparams.filter_length,
                                        hparams.hop_length, hparams.win_length,
                                        hparams.n_mel_channels,
                                        hparams.sampling_rate,
                                        hparams.mel_fmin, hparams.mel_fmax)

    def __getitem__(self, index):
        (audio, sample_rate, text, _, _, _) = self.dataset[index]
        assert self.stft.sampling_rate == sample_rate

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm,
                                             requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0)

        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return (text_norm, melspec)

    def __len__(self):
        return len(self.dataset)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(
            [len(x[0]) for x in batch]),
                                                          dim=0,
                                                          descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        mel_speaker = torch.FloatTensor(len(batch),num_mels,128)
        mel_speaker.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1:] = 1
            output_lengths[i] = mel.size(1)

            if mel.shape[1] <= 128:
                mel_slice = mel
            else:
                slice_start = random.randint(0,mel.shape[1]-128)
                mel_slice = mel[:,slice_start:slice_start+128]
            mel_speaker[i,:,:mel_slice.shape[1]] = mel_slice

        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, mel_speaker
