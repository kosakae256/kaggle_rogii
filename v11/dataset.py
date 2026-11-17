
import numpy as np
import torch
from torch.utils.data import Dataset

class WellSeqDataset(Dataset):
    """
    Dataset loader applying z-score scaling, windowing, 
    and padding masks to input well measurements.
    """
    def __init__(self, well_data_list, window_size=512, stride=256, mean=None, std=None):
        self.W = window_size
        self.stride = stride

        # Collect features to compute global stats if not provided
        all_feats = [d['features'] for d in well_data_list]
        if mean is None or std is None:
            cat = np.concatenate(all_feats, axis=0)
            self.mean = cat.mean(axis=0).astype(np.float32)
            self.std = np.maximum(cat.std(axis=0), 1e-6).astype(np.float32)
        else:
            self.mean = mean.astype(np.float32)
            self.std = std.astype(np.float32)

        self.windows = []
        for d in well_data_list:
            feat = (d['features'] - self.mean) / self.std
            target = d['target']
            md_s = d['md_since']
            L = len(feat)

            if L <= self.W:
                pad_len = self.W - L
                feat_pad = np.pad(feat, ((0, pad_len), (0, 0)), mode='constant', constant_values=0.)
                tgt_pad = np.pad(target, (0, pad_len), mode='constant', constant_values=0.)
                mask = np.zeros(self.W, dtype=np.float32); mask[:L] = 1.0
                pos_pad = np.pad(md_s, (0, pad_len), mode='constant', constant_values=0.)
                self.windows.append((feat_pad, tgt_pad, mask, pos_pad))
            else:
                starts = list(range(0, L - self.W + 1, self.stride))
                if starts[-1] + self.W < L: starts.append(L - self.W)
                for s in starts:
                    self.windows.append((
                        feat[s:s+self.W], target[s:s+self.W],
                        np.ones(self.W, dtype=np.float32), md_s[s:s+self.W]
                    ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        feat, tgt, mask, pos = self.windows[idx]
        return (
            torch.from_numpy(feat),
            torch.from_numpy(tgt),
            torch.from_numpy(mask),
            torch.from_numpy(pos),
        )
