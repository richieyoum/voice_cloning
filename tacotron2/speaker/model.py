#Reference: https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/0713f860a3dd41afb56e83cff84dbdf589d5e11a/encoder/model.py

from torch import nn
import numpy as np
import torch

class SpeakerEncoder(nn.Module):
    """ Learn speaker representation from speech utterance of arbitrary lengths.
    """
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device

        # lstm block consisting of 3 layers
        # takes input 40 channel log-mel spectrograms, projected to 256 dimensions
        self.lstm = nn.LSTM(
            input_size=40,
            hidden_size=256,
            num_layers=3,
            batch_first=True,
            dropout=0,
            bidirectional=False,
            proj_size=256
        ).to(device)

        self.linear = nn.Linear(in_features=256, out_features=256).to(device)
        self.relu = nn.ReLU().to(device)
        # epsilon term for numerical stability ( ie - division by 0)
        self.epsilon = 1e-5

    def forward(self, utterances, h_init=None, c_init=None):
        # implement section 2.1 from https://arxiv.org/pdf/1806.04558.pdf
        out, (hidden, cell) = self.lstm(utterances, (h_init, c_init))

        # compute speaker embedding from hidden state of final layer
        final_hidden = hidden[-1]
        speaker_embedding = self.relu(self.linear(final_hidden))

        # l2 norm of speaker embedding
        speaker_embedding = speaker_embedding / (torch.norm(speaker_embedding, dim=1, keepdim=True) + self.epsilon)
        return speaker_embedding

    def gradient_clipping():
        pass

    def similarity_matrix(self, embeds):
        pass

    def loss(self, embeds):
        pass