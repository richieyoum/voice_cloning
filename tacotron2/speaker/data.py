import torch
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms
from collections import defaultdict
import random

class SpeakerMelLoader(torch.utils.data.Dataset):
    """
    computes mel-spectrograms from audio file and pulls the speaker ID from the
    dataset
    """

    def __init__(self, dataset, format='speaker', speaker_utterances=4, mel_length = 160):
        self.dataset = dataset
        self.set_format(format)
        self.speaker_utterances = speaker_utterances
        self.mel_length = mel_length

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

    def get_mel(self, waveform, sampling_rate):
        audio = waveform.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = transforms.MFCC(sample_rate=sampling_rate)(audio)
        melspec = torch.squeeze(melspec, 0)
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
                mel_frame = random.randint(0,mel.shape[2]-self.mel_length)
                mels.append(mel[:,:,mel_frame:mel_frame+self.mel_length])
            return (speaker_id, torch.cat(mels,0))
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.format == 'utterance':
            return len(self.dataset)
        elif self.format == 'speaker':
            return len(self.speaker_keys)
        else:
            raise NotImplementedError()