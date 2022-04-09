# https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/0713f860a3dd41afb56e83cff84dbdf589d5e11a/encoder/train.py

import torch
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms
from speaker.data import SpeakerMelLoader
from speaker.model import SpeakerEncoder

def load_data(directory=".", batch_size=2, format='speaker', utter_per_speaker = 4):
    loader = SpeakerMelLoader(datasets.LIBRISPEECH(directory, download=True), format, utter_per_speaker)
    return torch.utils.data.DataLoader(
        loader,
        batch_size,
        num_workers=8,
    )

def train(speaker_per_batch=4, utter_per_speaker=4, learning_rate=1e-4):
    # Init data loader
    loader = load_data(".")

    # Device
    # Loss calc may run faster on cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    # Init model
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for step, batch in enumerate(loader):
        embeds = model(inputs)

        #Forward
        embeds_for_loss = embeds.view((speaker_per_batch,utter_per_speaker,-1)).to(loss_device)
        loss = model.loss(embeds_for_loss)

        #Backward
        model.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model


if __name__ == '__main__':
    for speaker_id, mel in load_data():
        print(speaker_id, mel.shape)