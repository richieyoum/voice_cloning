import torch
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms
from speaker.data import SpeakerMelLoader
from speaker.model import SpeakerEncoder
from speaker.utils import get_mapping_array

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from matplotlib import pyplot as plt

import os
from os import path

import numpy as np

diagram_path = 'diagrams'
accuracy_path = 'accuracy'
loss_path = 'loss'
silhouette_path = 'silhouette'
tsne_path = 'tsne'


def load_data(directory=".", batch_size=4, format='speaker', utter_per_speaker = 4, mel_type='Tacotron'):
    dataset = SpeakerMelLoader(datasets.LIBRISPEECH(directory, download=True), format, utter_per_speaker,mel_type=mel_type)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        num_workers=4,
        shuffle=True
    )


def load_validation(directory=".", batch_size=4, format='speaker', utter_per_speaker = 4, mel_type='Tacotron'):
    dataset = SpeakerMelLoader(datasets.LIBRISPEECH(directory, "dev-clean",download=True), format, utter_per_speaker,mel_type=mel_type)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        num_workers=4,
        shuffle=True
    )


def train(speaker_per_batch=4, utter_per_speaker=4, epochs=2, learning_rate=1e-4, mel_type='Tacotron'):
    # Init data loader
    train_loader = load_data(".", speaker_per_batch, 'speaker', utter_per_speaker,mel_type=mel_type)
    valid_loader = load_validation(".", speaker_per_batch, 'speaker', utter_per_speaker,mel_type=mel_type)

    # Device
    # Loss calc may run faster on cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    # Init model
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    sil_scores = np.zeros(0)
    gender_scores = np.zeros(0)
    val_losses = np.zeros(0)
    val_accuracy = np.zeros(0)

    gender_mapper = get_mapping_array()

    # Train loop
    for e in range(epochs):
        print('epoch:', e+1, 'of', epochs)

        model.train()
        # train_ids = np.zeros(0)
        # train_embeds = np.zeros((0, 256))
        for step, batch in enumerate(train_loader):
            #Forward
            #inputs: (speaker, utter, mel_len, mel_channel)
            speaker_id, inputs = batch
            #embed_inputs: (speaker*utter, mel_len, mel_channel)
            embed_inputs = inputs.reshape(-1, *(inputs.shape[2:])).to(device)
            #embeds: (speaker*utter, embed_dim)
            embeds = model(embed_inputs)
            #loss_embeds: (speaker, utter, embed_dim)
            loss_embeds = embeds.view((speaker_per_batch,utter_per_speaker,-1)).to(loss_device)
            loss = model.softmax_loss(loss_embeds)

            if step % 10 == 0:
                print('train e{}-s{}:'.format(e + 1, step + 1), 'loss', loss)

            #Backward
            model.zero_grad()
            loss.backward()
            model.gradient_clipping()
            optimizer.step()

            # train_ids = np.concatenate((train_ids, np.repeat(speaker_id, inputs.shape[1])))
            # train_embeds = np.concatenate((train_embeds, embeds))
        
        model.eval()
        loss = 0
        acc = 0

        valid_ids = np.zeros(0)
        valid_embeds = np.zeros((0, 256))

        for step,batch in enumerate(valid_loader):
            with torch.no_grad():
                speaker_id, inputs = batch
                embed_inputs = inputs.reshape(-1, *(inputs.shape[2:])).to(device)
                embeds = model(embed_inputs)
                loss_embeds = embeds.view((speaker_per_batch,utter_per_speaker,-1)).to(loss_device)
                loss += model.softmax_loss(loss_embeds)
                acc += model.accuracy(loss_embeds)
                valid_ids = np.concatenate((valid_ids, np.repeat(speaker_id, inputs.shape[1])))
                valid_embeds = np.concatenate((valid_embeds, embeds.to(loss_device).detach()))

        val_losses = np.concatenate((val_losses, [loss.to(loss_device).detach() / (step + 1)]))
        val_accuracy = np.concatenate((val_accuracy, [acc.to(loss_device).detach() / (step + 1)]))
        sil_scores = np.concatenate((sil_scores, [silhouette_score(valid_embeds, valid_ids)]))
        gender_scores = np.concatenate((gender_scores, [silhouette_score(valid_embeds, gender_mapper[valid_ids.astype('int')])]))
        print('valid e{}'.format(e + 1), 'loss', val_losses[-1])
        print('valid e{}'.format(e + 1), 'accuracy', val_accuracy[-1])
        print('silhouette score', sil_scores[-1])
        print('gender silhouette score', gender_scores[-1])

        plot_speaker_embeddings(valid_embeds, valid_ids, f'tsne_e{e + 1}_speaker.png', f'T-SNE Plot: Epoch {e + 1}')
        plot_random_embeddings(valid_embeds, valid_ids, f'tsne_e{e + 1}_random.png', title=f'T-SNE Plot: Epoch {e + 1}')
        plot_gender_embeddings(valid_embeds, valid_ids, f'tsne_e{e + 1}_gender.png', f'T-SNE Plot: Epoch {e + 1}')

        save_model(model, path.join('speaker', f'saved_model_e{e + 1}.pt'))

        plt.figure()
        plt.title('Silhouette Scores')
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.plot(np.arange(e + 1) + 1, sil_scores)
        # plt.show()
        plt.savefig(path.join(diagram_path, silhouette_path, f'sil_scores_{e + 1}.png'))
        plt.close()

        plt.figure()
        plt.title('Silhouette Scores over Gender')
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.plot(np.arange(e + 1) + 1, gender_scores)
        # plt.show()
        plt.savefig(path.join(diagram_path, silhouette_path, f'gender_scores_{e + 1}.png'))
        plt.close()

        plt.figure()
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(np.arange(e + 1) + 1, val_losses)
        # plt.show()
        plt.savefig(path.join(diagram_path, loss_path, f'val_losses_{e + 1}.png'))
        plt.close()

        plt.figure()
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(np.arange(e + 1) + 1, val_accuracy)
        # plt.show()
        plt.savefig(path.join(diagram_path, accuracy_path, f'val_accuracy_{e + 1}.png'))
        plt.close()
        
    return model


def save_model(model, path):
    #Save model state to path
    torch.save(model.state_dict(),path)


def load_model(path, device = None):
    #Instantiate Model
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")
    model = SpeakerEncoder(device, loss_device)

    #Load model state
    model.load_state_dict(torch.load(path))
    # Try this if running on multi-gpu setup or running model on cpu
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-across-devices
    # model.load_state_dict(torch.load(PATH, map_location=device))
    return model


def check_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    print('**loading model')
    model = load_model(path)

    print('**loading data')
    # data = load_data()
    data = load_validation()

    print('**running model')
    loss_total = 0
    acc_total = 0
    all_ids = np.zeros(0)
    all_embeds = np.zeros((0, 256))

    for step, batch in enumerate(data):
        speaker_id, inputs = batch

        print('batch:', step)
        embed_inputs = inputs.reshape(-1, *(inputs.shape[2:])).to(device)
        embeds = model(embed_inputs)
        loss_embeds = embeds.view(*(inputs.shape[:2]),-1).to(loss_device)
        loss = model.softmax_loss(loss_embeds)
        accuracy = model.accuracy(loss_embeds)

        all_ids = np.concatenate((all_ids, np.repeat(speaker_id, inputs.shape[1])))
        all_embeds = np.concatenate((all_embeds, embeds.to(loss_device).detach()))

        loss_total += loss
        acc_total += accuracy
        
        # print('inputs.shape',inputs.shape)
        # print('embed_inputs.embed_inputs',embeds.shape)
        # print('embeds.shape',embeds.shape)
        # print('loss_embeds.shape',loss_embeds.shape)
        # print('loss.shape',loss.shape)
        # print('loss',loss)
        # print('accuracy',accuracy)
    
    print('average loss', loss_total / (step+1))
    print('average accuracy', acc_total / (step+1))
    print('silhouette score', silhouette_score(all_embeds, all_ids))
    plot_speaker_embeddings(all_embeds, all_ids, path.join(diagram_path, 'tsne_speaker_saved_model.png'))
    plot_random_embeddings(all_embeds, all_ids, path.join(diagram_path, 'tsne_speaker_saved_model.png'))
    plot_gender_embeddings(all_embeds, all_ids, path.join(diagram_path, 'tsne_gender_saved_model.png'))


def plot_gender_embeddings(embeddings, ids, filename, title='T-SNE Plot'):
    # Per https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # reducing dimensionality before running TSNE
    pca = PCA(50)
    reduction = pca.fit_transform(embeddings)
    tsne = TSNE(init='pca', learning_rate='auto')
    transformed = tsne.fit_transform(reduction)

    gender_mapper = get_mapping_array()
    genders = gender_mapper[ids.astype('int')]
    females = genders == 1
    males = genders == 2

    plt.figure()
    plt.title(title)

    plt.scatter(transformed[females, 0], transformed[females, 1], label='Female')
    plt.scatter(transformed[males, 0], transformed[males, 1], label='Male')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(path.join(diagram_path, tsne_path, filename))
    plt.close()


def plot_speaker_embeddings(embeddings, ids, filename, title='T-SNE Plot'):
    # Per https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # reducing dimensionality before running TSNE
    pca = PCA(50)
    reduction = pca.fit_transform(embeddings)
    tsne = TSNE(init='pca', learning_rate='auto')
    transformed = tsne.fit_transform(reduction)

    ids = ids.astype('int')
    unique_ids = np.unique(ids)

    plt.figure()
    plt.title(f'{title} Speakers')

    for speaker_id in unique_ids:
        speaker_idx = ids == speaker_id
        plt.scatter(transformed[speaker_idx, 0], transformed[speaker_idx, 1], label=f'Speaker {speaker_id}')

    # plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(path.join(diagram_path, tsne_path, filename))
    plt.close()


def plot_random_embeddings(embeddings, ids, filename, size=15, title='T-SNE Plot Random'):
    # Per https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # reducing dimensionality before running TSNE
    pca = PCA(50)
    reduction = pca.fit_transform(embeddings)
    tsne = TSNE(init='pca', learning_rate='auto')
    transformed = tsne.fit_transform(reduction)

    ids = ids.astype('int')
    unique_ids = np.unique(ids)
    random_unique_ids = np.random.choice(ids, size=min(size, unique_ids.size), replace=False)

    plt.figure()

    plt.title(f'{title} - {random_unique_ids.size} Speakers')

    for speaker_id in random_unique_ids:
        speaker_idx = ids == speaker_id
        plt.scatter(transformed[speaker_idx, 0], transformed[speaker_idx, 1], label=f'Speaker {speaker_id}')

    # plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(path.join(diagram_path, tsne_path, filename))
    plt.close()


if __name__ == '__main__':
    os.makedirs(diagram_path, exist_ok=True)
    os.makedirs(path.join(diagram_path, loss_path), exist_ok=True)
    os.makedirs(path.join(diagram_path, accuracy_path), exist_ok=True)
    os.makedirs(path.join(diagram_path, tsne_path), exist_ok=True)
    os.makedirs(path.join(diagram_path, silhouette_path), exist_ok=True)
    # for speaker_id, mel in load_data():
    #     print(speaker_id, mel.shape)

    # Might make sense to adjust speaker / utterance per batch, e.g. 64/10    
    m = train(epochs=1000)

    # save_model(m,'speaker/saved_model.pt')
    # check_model('speaker/saved_model.pt')
