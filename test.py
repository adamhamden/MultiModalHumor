from data import Data
import torch.optim as optim
from MultiModalHumor.model import *
from config import config, humor_speakers, speakers
from sklearn.metrics import confusion_matrix

config = config()
common_kwargs = dict(path2data='../PATS/data',
                     speaker=['fallon', 'rock'],
                     modalities=['pose/normalize','audio/log_mel_512', 'text/bert'],
                     fs_new=[15, 15, 15],
                     time=4.3,
                     batch_size=config['batch_size'],
                     window_hop=5,
                     shuffle=True)

model = HumorClassifier(config).to(config['device'])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

for epoch in range(config['epochs']):
    data = Data(**common_kwargs)
    style_dict = data.style_dict
    style_to_speaker_dict = {v: k for k, v in style_dict.items()}
    print(style_to_speaker_dict)
    train_loss = 0.0
    i = 0
    model.train()
    for batch in data.train:
        x_t = batch['text/bert']
        x_a = batch['audio/log_mel_512']
        x_p = batch['pose/normalize']

        x_t = x_t.to(config['device'])
        x_a = x_a.to(config['device'])
        x_p = x_p.to(config['device'])

        # this is assuming time*fs = 64
        if x_t.shape[0] != config['batch_size']:
            break

        x_t = x_t.reshape((config['batch_size'], config['context_length'], -1, config['lstm_text_input']))
        x_a = x_a.reshape((config['batch_size'], config['context_length'], -1, config['lstm_audio_input']))
        x_p = x_p.reshape((config['batch_size'], config['context_length'], -1, config['lstm_pose_input']))

        styles = batch['style'][:, 0]
        #print(batch['style'])
        speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
        #print(speakers)
        batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]
        #print(f'1s = {batch_label.count(1)} and 0s = {batch_label.count(0)}')
        batch_label = torch.Tensor(batch_label).unsqueeze(1).to(config['device'])

        optimizer.zero_grad()
        pred = model(x_t.float(), x_a.float(), x_p.float()).squeeze(0)

        loss = criterion(pred, batch_label.float())

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch} loss: {train_loss/len(data.train):.3f}')

torch.save(model.state_dict(), './trained_models/model_2')