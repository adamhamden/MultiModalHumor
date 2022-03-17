from data import Data
from tqdm import tqdm
import torch

common_kwargs = dict(path2data = '../PATS/data',
                     speaker = ['bee', 'rock' ],
                     modalities = ['pose/data',],
                     fs_new = [15, 15, 15],
                     batch_size = 4,
                     window_hop = 5)

data = Data(**common_kwargs)
style_dict = data.style_dict
style_to_speaker_dict = {v:k for k, v in style_dict.items()}

humor_speakers = ['oliver', #TV sitting high_freq
                  'jon', #TV sitting
                  'conan', #TV standing high_freq
                  'ellen', #TV standing
                  'seth', #TV sitting low frequency
                  'colbert', #TV standing high_freq
                  'corden', #TV standing
                  'fallon', #TV standing
                  'huckabee', #TV standing
                  'maher', #TV standing
                  'minhaj', #TV standing
                  'bee', #TV standing
                  'noah' #TV sitting
                 ]

for batch in data.train:
    pose = batch['pose/data']

    styles = batch['style'][:,0]
    speakers = list(map(lambda x: style_to_speaker_dict[x], styles.numpy()))
    batch_label = [1 if speaker in humor_speakers else 0 for speaker in speakers]
    batch_label = torch.Tensor(batch_label)

    break