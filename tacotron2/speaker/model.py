#Reference: https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/0713f860a3dd41afb56e83cff84dbdf589d5e11a/encoder/model.py

from torch import nn
import numpy as np
import torch

class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        pass

    def forward(self, utterances, hidden_init=None):
        pass

    def gradient_clipping():
        pass

    def similarity_matrix(self, embeds):
        pass

    def loss(self, embeds):
        pass