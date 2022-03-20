from data import Data
import torch
import torch.nn as nn
import torch.optim as optim
from config import config, humor_speakers, speakers
from sklearn.metrics import confusion_matrix


class LogRegClassifier(nn.Module):
    def __init__(self):
        super(LogRegClassifier, self).__init__()

        self.l1 = nn.Linear(1, 32)
        self.l2 = nn.Linear(32, 8)
        self.l3 = nn.Linear(8,1)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):

        out = self.dropout(torch.relu(self.l1(x)))
        out = self.dropout(torch.relu(self.l2(out)))
        out = self.l3(out)
        return out


class Average():
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.avg = None

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = None

    def update(self, value):
        self.sum += value
        self.count += 1
        self.avg = self.sum/self.count


def get_pose_std_dev(pose):
    # [time, 104] Tensor
    return float(pose.std(dim=0).sum())


device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = .01
epochs = 4

model = LogRegClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

common_kwargs = dict(path2data='../PATS/data',
                     speaker=['rock', 'seth',],
                     modalities=['pose/normalize'],
                     fs_new=[15, 15, 15],
                     time=4.3,
                     batch_size=4,
                     window_hop=5,
                     shuffle=True)

for epoch in range(epochs):
    data = Data(**common_kwargs)
    style_dict = data.style_dict
    style_to_speaker_dict = {v: k for k, v in style_dict.items()}
    train_loss = 0.0
    model.train()
    for batch in data.train:
        x_p = batch['pose/normalize']
        x_p = x_p.to(device)
        batch_size = x_p.shape[0]

        x_p = torch.Tensor(list(map(get_pose_std_dev, x_p))).reshape(batch_size, -1).to(device)

        styles = batch['style'][:, 0]
        speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
        batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]
        batch_label = torch.Tensor(batch_label).unsqueeze(1).to(device)

        optimizer.zero_grad()
        pred = model(x_p.float()).squeeze(0)

        loss = criterion(pred, batch_label.float())

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch} loss: {train_loss/len(data.train):.3f}')



"""common_kwargs = dict(path2data='../PATS/data',
                     speaker=['seth'],
                     modalities=['pose/normalize'],
                     fs_new=[15, 15, 15],
                     time=4.3,
                     batch_size=4,
                     window_hop=5,
                     shuffle=True)

data = Data(**common_kwargs)

pose_avg = Average()

for batch in data.train:
    x_p = batch['pose/normalize']

    pose_stds = list(map(get_pose_std_dev, x_p))
    list(map(pose_avg.update, pose_stds))

print(pose_avg.avg)
"""

model.eval()
y_pred = []
y_true = []

for batch in data.test:
    x_p = batch['pose/normalize']

    batch_size = x_p.shape[0]

    x_p = x_p.to(device)
    x_p = x_p.reshape(batch_size, -1)

    styles = batch['style'][:, 0]
    speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
    batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]
    y_true.extend(batch_label)

    pred = model(x_p.float()).squeeze(0)
    pred = torch.sigmoid(pred)
    pred = pred.reshape(-1)
    pred = (pred > .5).float()
    y_pred.extend(pred.tolist())

cm = confusion_matrix(y_true, y_pred)
#sn.heatmap(cm, annot=True)
#plt.savefig('output.png')
print(cm)