from data import Data
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from model import *

humor_speakers = ['oliver',  # TV sitting high_freq
                  'jon',  # TV sitting
                  'conan',  # TV standing high_freq
                  'ellen',  # TV standing
                  'seth',  # TV sitting low frequency
                  'colbert',  # TV standing high_freq
                  'corden',  # TV standing
                  'fallon',  # TV standing
                  'huckabee',  # TV standing
                  'maher',  # TV standing
                  'minhaj',  # TV standing
                  'bee',  # TV standing
                  'noah'  # TV sitting
                  ]

config = dict(context_length=4,
              batch_size= 4,
              classes=2,
              learning_rate=.01,
              epochs=10,
              unimodal_context=dict(text_lstm_input=768,
                                    audio_lstm_input=128,
                                    pose_lstm_input=104,
                                    text_lstm_hidden_size=128,
                                    audio_lstm_hidden_size=32,
                                    pose_lstm_hidden_size=32
                                    ),
              use_text=True,
              use_audio=True,
              use_pose=True,
              device='cpu',
              multimodal_context=dict(text_dropout=.1,
                                      audio_dropout=.1,
                                      pose_dropout=.1,
                                      text_input=128*4,
                                      audio_input=32*4,
                                      pose_input=32*4,
                                      text_mmcn_hidden_size=128,
                                      audio_mmcn_hidden_size=32,
                                      pose_mmcn_hidden_size=32,
                                      d_input=192,
                                      d_target=192,
                                      max_length=1000,
                                      d_model=512,
                                      n_head=6,
                                      n_layers=4,
                                      d_feedforward=256,
                                      d_key=192,
                                      d_value=192,
                                      dropout=.1
                                      ),
              )


common_kwargs = dict(path2data='../PATS/data',
                     speaker=['bee'],
                     modalities=['pose/data','audio/log_mel_512'],
                     fs_new=[15, 15, 15],
                     batch_size=config['batch_size'],
                     window_hop=5)

data = Data(**common_kwargs)
style_dict = data.style_dict
style_to_speaker_dict = {v: k for k, v in style_dict.items()}

model = HumorClassifier(config).to(config['device'])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


for epoch in range(config['epochs']):
    epoch_loss = 0.0
    i = 0
    for batch in data.train:
        x_t = torch.zeros((config['batch_size'], 64, 768))#batch['text/bert']
        x_a = batch['audio/log_mel_512']
        x_p = batch['pose/data']

        x_t.to(config['device'])
        x_a.to(config['device'])
        x_p.to(config['device'])

        x_t = x_t.reshape((config['batch_size'], config['context_length'], -1, config['unimodal_context']['text_lstm_input']))
        x_a = x_a.reshape((config['batch_size'], config['context_length'], -1, config['unimodal_context']['audio_lstm_input']))
        x_p = x_p.reshape((config['batch_size'], config['context_length'], -1, config['unimodal_context']['pose_lstm_input']))

        styles = batch['style'][:, 0]
        speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
        batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]
        batch_label = torch.Tensor(batch_label)

        optimizer.zero_grad()
        pred = model(x_t.float(), x_a.float(), x_p.float()).squeeze(0)
        loss = criterion(pred, batch_label.float())

        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        i+=1
        if i%200 == 0:
            print(f'Epoch {epoch} loss: {epoch_loss / 200:.3f}')
        break
