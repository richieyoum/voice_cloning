# https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/0713f860a3dd41afb56e83cff84dbdf589d5e11a/encoder/train.py

import torch
import torchaudio.datasets as datasets
import torchaudio.transforms as transforms
from speaker.data import SpeakerMelLoader
from speaker.model import SpeakerEncoder

def load_data(directory=".", batch_size=4, format='speaker', utter_per_speaker = 4):
    dataset = SpeakerMelLoader(datasets.LIBRISPEECH(directory, download=True), format, utter_per_speaker)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        num_workers=4,
        shuffle=True
    )

def load_validation(directory=".", batch_size=4, format='speaker', utter_per_speaker = 4):
    dataset = SpeakerMelLoader(datasets.LIBRISPEECH(directory, "dev-clean",download=True), format, utter_per_speaker)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        num_workers=4,
        shuffle=True
    )

def train(speaker_per_batch=4, utter_per_speaker=4, epochs=2, learning_rate=1e-4):
    # Init data loader
    train_loader = load_data(".", speaker_per_batch, 'speaker', utter_per_speaker)
    valid_loader = load_validation(".", speaker_per_batch, 'speaker', utter_per_speaker)

    # Device
    # Loss calc may run faster on cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_device = torch.device("cpu")

    # Init model
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train loop
    for e in range(epochs):
        print('epoch:',e+1,'of',epochs)

        model.train() 
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
                print('train e{}-s{}:'.format(e,step),'loss',loss)

            #Backward
            model.zero_grad()
            loss.backward()
            model.gradient_clipping()
            optimizer.step()
        
        model.eval()
        loss = 0
        acc = 0
        for step,batch in enumerate(valid_loader):
            with torch.no_grad():
                speaker_id, inputs = batch
                embed_inputs = inputs.reshape(-1, *(inputs.shape[2:])).to(device)
                embeds = model(embed_inputs)
                loss_embeds = embeds.view((speaker_per_batch,utter_per_speaker,-1)).to(loss_device)
                loss += model.softmax_loss(loss_embeds)
                acc += model.accuracy(loss_embeds)

        print('valid e{}'.format(e), 'loss', loss/(step+1))
        print('valid e{}'.format(e), 'accuracy', acc/(step+1))
        
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

    for step, batch in enumerate(data):
        speaker_id, inputs = batch

        print('batch:', step)
        embed_inputs = inputs.reshape(-1, *(inputs.shape[2:])).to(device)
        embeds = model(embed_inputs)
        loss_embeds = embeds.view(*(inputs.shape[:2]),-1).to(loss_device)
        loss = model.softmax_loss(loss_embeds)
        accuracy = model.accuracy(loss_embeds)

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


if __name__ == '__main__':
    # for speaker_id, mel in load_data():
    #     print(speaker_id, mel.shape)
    
    # m = train(epochs=10)
    # save_model(m,'speaker/saved_model.pt')

    check_model('speaker/saved_model.pt')