from data import Data
from tqdm import tqdm

common_kwargs = dict(path2data = '../PATS/data',
                     speaker = ['bee'],
                     modalities = ['pose/data',],
                     fs_new = [15, 15, 15],
                     batch_size = 4,
                     window_hop = 5)

data = Data(**common_kwargs)

for batch in data.train:
    pose = batch['pose/data']
    print(pose.shape)
    print("something")

    break