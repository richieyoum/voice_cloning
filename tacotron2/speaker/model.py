#Reference: https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/0713f860a3dd41afb56e83cff84dbdf589d5e11a/encoder/model.py

from torch import nn
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

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
            bidirectional=False
        ).to(device)

        self.linear = nn.Linear(in_features=256, out_features=256).to(device)
        self.relu = nn.ReLU().to(device)
        # epsilon term for numerical stability ( ie - division by 0)
        self.epsilon = 1e-5

        #Cosine similarity weights
        self.sim_weight = nn.Parameter(torch.tensor([5.])).to(loss_device)
        self.sim_bias = nn.Parameter(torch.tensor([-1.])).to(loss_device)

    def forward(self, utterances, h_init=None, c_init=None):
        # implement section 2.1 from https://arxiv.org/pdf/1806.04558.pdf
        if h_init is None or c_init is None:
            out, (hidden, cell) = self.lstm(utterances)
        else:
            out, (hidden, cell) = self.lstm(utterances, (h_init, c_init))

        # compute speaker embedding from hidden state of final layer
        final_hidden = hidden[-1]
        speaker_embedding = self.relu(self.linear(final_hidden))

        # l2 norm of speaker embedding
        speaker_embedding = speaker_embedding / (torch.norm(speaker_embedding, dim=1, keepdim=True) + self.epsilon)
        return speaker_embedding

    def gradient_clipping(self):
        self.sim_weight.grad *= 0.01
        self.sim_bias.grad *= 0.01

        #Pytorch to clip gradients if norm greater than max
        clip_grad_norm_(self.parameters(),max_norm=3,norm_type=2)

    def similarity_matrix(self, embeds, debug=False):
        # calculate s_ji,k from section 2.1 of GE2E paper
        # output matrix is cosine similarity between each utterance x centroid of each speaker
        # embeds input size: (speakers, utterances, embedding size)

        # Speaker centroids
        # Equal to average of utterance embeddings for the speaker
        # Used for neg examples (utterance comparing to false speaker)
        # Equation 1 in paper
        # size: (speakers, 1, embedding size)
        speaker_centroid = torch.mean(embeds,dim=1,keepdim=True)

        # Utterance exclusive centroids
        # Equal to average of utterance embeddings for the speaker, excluding ith utterance
        # Used for pos samples (utterance comparing to true speaker; speaker centroid exludes the utterance)
        # Equation 8 in paper
        # size: (speakers, utterances, embedding size)
        num_utterance = embeds.shape[1]
        utter_ex_centroid = (torch.sum(embeds,dim=1,keepdim=True) - embeds) / (num_utterance-1)

        if debug:
            print("e",embeds.shape)
            print(embeds)
            print("sc",speaker_centroid.shape)
            print(speaker_centroid)
            print("uc",utter_ex_centroid.shape)
            print(utter_ex_centroid)

        # Create pos and neg masks
        num_speaker = embeds.shape[0]
        i = torch.eye(num_speaker, dtype=torch.int)
        pos_mask = torch.where(i)
        neg_mask = torch.where(1-i)

        if debug:
            print("pm",len(pos_mask),len(pos_mask[0]))
            print(pos_mask)
            print("nm",len(neg_mask),len(neg_mask[0]))
            print(neg_mask)

        # Compile similarity matrix
        # size: (speakers, utterances, speakers)
        # initial size is (speakers, speakers, utterances for easier vectorization)
        sim_matrix = torch.zeros(num_speaker, num_speaker, num_utterance).to(self.loss_device)
        sim_matrix[pos_mask] = nn.functional.cosine_similarity(embeds,utter_ex_centroid,dim=2)
        sim_matrix[neg_mask] = nn.functional.cosine_similarity(embeds[neg_mask[0]],speaker_centroid[neg_mask[1]],dim=2)
        if debug:
            print("sm",sim_matrix.shape)
            print("pos vals",sim_matrix[pos_mask])
            print("neg vals",sim_matrix[neg_mask])
            print(sim_matrix)
        
        sim_matrix = sim_matrix.permute(0,2,1)

        if debug:
            print("sm",sim_matrix.shape)
            print(sim_matrix)
            print("cos sim weight", sim_weight)
            print("cos sim bias", sim_bias)

        # Apply weight / bias
        sim_matrix = sim_matrix * self.sim_weight + self.sim_bias
        return sim_matrix

    def softmax_loss(self, embeds):
        """
        computes softmax loss as defined by equ 6 in the GE2E paper
        :param embeds: shape (speakers, utterances, embedding size)
        :return: computed softmax loss
        """
        # per the GE2E paper, softmax loss as defined by equ 6
        # performs slightly better over Text-Independent Speaker
        # Verification tasks.
        # ref section 2.1 of the GE2E paper
        speaker_count = embeds.shape[0]

        # speaker, utterance, speaker
        similarities = self.similarity_matrix(embeds)

        # equ 6
        loss_matrix = -similarities[torch.arange(0, speaker_count), :, torch.arange(0, speaker_count)] + \
                      torch.log(torch.sum(torch.exp(similarities), dim=2))

        # equ 10
        return torch.sum(loss_matrix)

    def contrast_loss(self, embeds):
        """
        computes contrast loss as defined by equ 7 in the GE2E paper
        :param embeds: shape (speakers, utterances, embedding size)
        :return: computed softmax loss
        """
        # per the GE2E paper, contrast loss as defined by equ 7
        # performs slightly better over Text-Dependent Speaker
        # Verification tasks.
        # ref section 2.1 of the GE2E paper
        speaker_count, utterance_count = embeds.shape[0:2]

        # speaker, utterance, speaker
        similarities = self.similarity_matrix(embeds)

        # Janky indexing to resolve k != j
        mask = torch.ones(similarities.shape, dtype=torch.bool)
        mask[torch.arange(speaker_count), :, torch.arange(speaker_count)] = False
        closest_neighbors, _ = torch.max(similarities[mask].reshape(speaker_count, utterance_count, speaker_count - 1), dim=2)

        # Positive influence over matching embeddings
        matching_embedding = similarities[torch.arange(0, speaker_count), :, torch.arange(0, speaker_count)]

        # equ 7
        loss_matrix = 1 - torch.sigmoid(matching_embedding) + torch.sigmoid(closest_neighbors)

        # equ 10
        return torch.sum(loss_matrix)

