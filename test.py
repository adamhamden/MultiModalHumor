from data.data_utils import Data
from tqdm import tqdm

"""common_kwargs = dict(path2data = './data/PATS',
                     speaker = ['oliver'],
                     modalities = ['pose/data', 'audio/log_mel_512', 'text/bert'],
                     fs_new = [15, 15, 15],
                     batch_size = 4,
                     window_hop = 5)

data = Data(**common_kwargs)"""

from data.data_utils import HDF5

h5 = HDF5.h5_open('./data/PATS/processed/oliver/cmu0000006502.h5', 'r')
print(h5.keys())
for key in h5.keys():
    print('{}: {}'.format(key, h5[key].keys()))
h5.close()