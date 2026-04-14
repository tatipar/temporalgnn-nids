import os
import torch
from torch_geometric.data import Dataset


class NF_IDS_Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        # root_dir: (eg: "./dataset_processed")
        # split: 'train', 'val' or 'test'
        self.split_dir = os.path.join(root_dir, split)
        super().__init__(self.split_dir)

        # List files ordered numerically to respect the time
        self.files = sorted(
            [f for f in os.listdir(self.split_dir) if f.endswith('.pt')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )

    def len(self):
        return len(self.files)

    def get(self, idx):
        data = torch.load(
            os.path.join(self.split_dir, self.files[idx]),
            weights_only=False  # to allow loading complex graph objects
        )
        return data
