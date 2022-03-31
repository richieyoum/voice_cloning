# https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/0713f860a3dd41afb56e83cff84dbdf589d5e11a/encoder/train.py

import torch
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms


class SpeakerMelLoader(torch.utils.data.Dataset):
    """
    computes mel-spectrograms from audio file and pulls the speaker ID from the
    dataset
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def get_mel(self, waveform, sampling_rate):
        audio = waveform.unsqueeze(0)
        audio = torch.autograd.Variable(audio, requires_grad=False)
        melspec = transforms.MFCC(sample_rate=sampling_rate)(audio)
        melspec = torch.squeeze(melspec, 0)
        return melspec

    def __getitem__(self, index):
        (waveform, sample_rate, _, speaker_id, _, _) = self.dataset[index]
        mel = self.get_mel(waveform, sample_rate)
        return (speaker_id, mel)

    def __len__(self):
        return len(self. dataset)

def load_data(directory=".", batch_size=1):
    loader = SpeakerMelLoader(datasets.LIBRISPEECH(".", download=True))
    return torch.utils.data.DataLoader(
        loader,
        batch_size,
        num_workers=32,
    )

if __name__ == '__main__':
    for speaker_id, mel in load_data():
        print(speaker_id, mel.shape)
