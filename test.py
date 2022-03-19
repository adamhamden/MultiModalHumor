from data import Data
import torch.optim as optim
from MultiModalHumor.model import *
from config import config, humor_speakers

config = config()
common_kwargs = dict(path2data='../PATS/data',
                     speaker=['bee'],
                     modalities=['pose/data','audio/log_mel_512', 'text/bert'],
                     fs_new=[15, 15, 15],
                     time=4.3,
                     batch_size=config['batch_size'],
                     window_hop=5)

model = HumorClassifier(config).to(config['device'])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

for epoch in range(config['epochs']):
    data = Data(**common_kwargs)
    style_dict = data.style_dict
    style_to_speaker_dict = {v: k for k, v in style_dict.items()}

    epoch_loss = 0.0
    i = 0
    for batch in data.train:
        x_t = batch['text/bert']
        x_a = batch['audio/log_mel_512']
        x_p = batch['pose/data']

        x_t.to(config['device'])
        x_a.to(config['device'])
        x_p.to(config['device'])

        # this is assuming time*fs = 64
        if x_t.shape[0] != config['batch_size']:
            break

        x_t = x_t.reshape((config['batch_size'], config['context_length'], -1, config['lstm_text_input']))
        x_a = x_a.reshape((config['batch_size'], config['context_length'], -1, config['lstm_audio_input']))
        x_p = x_p.reshape((config['batch_size'], config['context_length'], -1, config['lstm_pose_input']))

        styles = batch['style'][:, 0]
        speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
        batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]
        batch_label = torch.Tensor(batch_label).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(x_t.float(), x_a.float(), x_p.float()).squeeze(0)

        loss = criterion(pred, batch_label.float())

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        i+=1
        if i%20 == 0:
            print(f'Epoch {epoch} loss: {epoch_loss / 20:.3f}')
