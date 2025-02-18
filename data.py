import numpy as np

import torch
from torch.utils.data import Dataset

from generate_training_data import make_img, add_noise




class myDataset(Dataset):
    def __init__(self, device, N=1000, transform=None):
        super().__init__()
        self.images = []
        self.targets = []
        self.device = device

        for ii in range(N):
            img, fine_img = make_img(np.random.randint(0, 32), 64)

            img, nimg = add_noise(img, fine_img, amount=np.random.uniform(1, 3))

            self.targets.append(img)
            self.images.append(nimg)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # here adding a dummy dimension for greyscale images
        #  would need to be adapted for RBG (or otherwise multiple inputs)
        return torch.tensor(self.images[idx], dtype=torch.float).to(self.device)[None,:,:], torch.tensor(self.targets[idx], dtype=torch.float).to(self.device)[None,:,:]

