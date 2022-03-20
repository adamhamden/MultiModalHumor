from data import Data
import torch.optim as optim
from MultiModalHumor.model import *
from config import config, humor_speakers, speakers
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sn
import matplotlib.pyplot as plt

config = config()
common_kwargs = dict(path2data='../PATS/data',
                     speaker=['fallon', 'rock'],
                     modalities=['pose/normalize','audio/log_mel_512', 'text/bert'],
                     fs_new=[15, 15, 15],
                     time=4.3,
                     batch_size=config['batch_size'],
                     window_hop=5)

model = HumorClassifier(config).to(config['device'])
model.load_state_dict(torch.load('./trained_models/model_1'))

model.eval()

data = Data(**common_kwargs)
style_dict = data.style_dict
style_to_speaker_dict = {v: k for k, v in style_dict.items()}

y_pred = []
y_true = []

for batch in data.test:
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
    speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
    batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]

    y_true.extend(batch_label)
    batch_label = torch.Tensor(batch_label).unsqueeze(1).to(config['device'])

    pred = model(x_t.float(), x_a.float(), x_p.float()).squeeze(0)
    pred = torch.sigmoid(pred)
    pred = pred.reshape(-1)
    pred = (pred > .5).float()
    y_pred.extend(pred.tolist())

cm = confusion_matrix(y_true, y_pred)
print(accuracy_score(y_true, y_pred))
#sn.heatmap(cm, annot=True)
#plt.savefig('output.png')
print(cm)