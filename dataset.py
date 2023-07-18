# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" Define the dance dataset. """
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset


def paired_collate_fn(insts):
    src_seq, tgt_seq = list(zip(*insts))
    src_pos = np.array([
        [pos_i + 1 for pos_i, v_i in enumerate(inst)] for inst in src_seq])

    src_seq = torch.FloatTensor(src_seq)
    src_pos = torch.LongTensor(src_pos)
    if tgt_seq[0] is None:
        tgt_seq = None
    else:
        tgt_seq = torch.FloatTensor(tgt_seq)

    return src_seq, src_pos, tgt_seq

class LightDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        d = self.data[index]
        l = d['lighting_array']
        a = d['music_array']
        if l is not None:
            return a, l
        else:
            return a

class PredictionDataset(Dataset):
    def __init__(self, music_data, light_data):
        self.m_data = music_data
        self.l_data = light_data

    def __len__(self):
        return len(self.m_data)

    def __getitem__(self, index):
        if self.l_data is not None:
            return self.m_data[index], self.l_data[index]
        else:
            return self.m_data[index], None