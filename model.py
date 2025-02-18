import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from visualize_results import *
from train_model import *
from generate_training_data import *
from data import *


class Denoise_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lin1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lin2 = nn.ReLU()
#        self.conv3 = nn.Conv2d(8, 4, kernel_size=3)
#        self.lin3 = nn.ReLU()
#        self.pool= nn.AvgPool2d(kernel_size=2, stride=2)
#        self.unpool= nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.lin2 = nn.ReLU()
        self.dconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.lin3 = nn.ReLU()
        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)


    def forward(self, x):
        #x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.lin1(x)
        xsave = x
        x = self.conv2(x)
        x = self.lin2(x)
        x = self.dconv1(x)
        x = self.lin2(x)
        x = self.dconv2(x)

        x += xsave
        x = self.lin3(x)

        return self.out(x)



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":

    device = get_device()

    NN = Denoise_Model()
    NN.to(device)

    # create training data
    BATCH_SIZE=1
    dataset = myDataset(device, N=1000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # create some more data for testing
    testdataset = myDataset(device)
    testdataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(NN.parameters(), lr=1e-4)
    train(NN, dataloader, testdataloader, 10, loss_fn, optimizer, device=device)


    # try model on example image
    test_image(NN, "./test_data/m33.fit", device=device)

