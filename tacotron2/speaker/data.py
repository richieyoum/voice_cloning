import torch
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms
from collections import defaultdict
import random
import layers

import warnings

class SpeakerMelLoader(torch.utils.data.Dataset):
    """
    computes mel-spectrograms from audio file and pulls the speaker ID from the
    dataset
    """

    def __init__(self, dataset, format='speaker', speaker_utterances=4, mel_length = 100, mel_type = 'Tacotron'):
        self.dataset = dataset
        self.set_format(format)
        self.speaker_utterances = speaker_utterances
        self.mel_length = mel_length
        self.mel_type = mel_type
        self.mel_generators = dict()

    def set_format(self,format):
        self.format = format

        if format == 'speaker':
            self.create_speaker_index()

    def create_speaker_index(self):
        vals = [x.split('-',1) for x in self.dataset._walker]
        speaker_map = defaultdict(list)

        for i,v in enumerate(vals):
            speaker_map[v[0]].append(i)
        
        self.speaker_map = speaker_map
        self.speaker_keys = list(speaker_map.keys())

    def get_mel_gen(self, sampling_rate, channels=80):
        if (sampling_rate, channels) not in self.mel_generators:
            if self.mel_type == 'MFCC':
                mel_gen = transforms.MFCC(sample_rate=sampling_rate, n_mfcc=channels)
            elif self.mel_type == 'Mel':
                mel_gen = transforms.MelSpectrogram(sample_rate=sampling_rate, n_mels=channels)
            elif self.mel_type == 'Tacotron':
                mel_gen = layers.TacotronSTFT(sampling_rate=sampling_rate,n_mel_channels=channels)
            else:
                raise NotImplemented
            self.mel_generators[(sampling_rate,channels)] = mel_gen
        return self.mel_generators[(sampling_rate, channels)]

    def get_mel(self, waveform, sampling_rate, channels=80):
        # We previously identified that these warnings were ok.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message=r'At least one mel filterbank has all zero values.*', module=r'torchaudio.*')
            audio = waveform.unsqueeze(0)
            audio = torch.autograd.Variable(audio, requires_grad=False)
            melspec = self.get_mel_gen(sampling_rate, channels)(audio)
            # melspec is (1,1,channels, time) by default
            # return (time, channels)
            melspec = torch.squeeze(melspec).T
            return melspec

    def __getitem__(self, index):
        if self.format == 'utterance':
            (waveform, sample_rate, _, speaker_id, _, _) = self.dataset[index]
            mel = self.get_mel(waveform, sample_rate)
            return (speaker_id, mel)
        elif self.format == 'speaker':
            speaker_id = self.speaker_keys[index]
            utter_indexes = random.sample(self.speaker_map[speaker_id], self.speaker_utterances)
            mels = []
            for i in utter_indexes:
                (waveform, sample_rate, _, speaker_id, _, _) = self.dataset[i]
                mel = self.get_mel(waveform, sample_rate)
                mel_frame = random.randint(0,mel.shape[0]-self.mel_length)
                mels.append(mel[mel_frame:mel_frame+self.mel_length,:])
            return (speaker_id, torch.stack(mels,0))
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.format == 'utterance':
            return len(self.dataset)
        elif self.format == 'speaker':
            return len(self.speaker_keys)
        else:
            raise NotImplementedError()