import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from glob import glob

from visualize_results import *
from train_model import *
from generate_training_data import *
from data import *


class Denoise_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lin1 = nn.GELU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lin2 = nn.GELU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lin3 = nn.GELU()


        self.dconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.lin4 = nn.GELU()
        self.dconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.lin5 = nn.GELU()
        self.dconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.lin6 = nn.GELU()
        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)


    def forward(self, x):
        #x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.lin1(x)
        xsave = x
        x = self.conv2(x)
        x = self.lin2(x)
        x = self.conv3(x)
        x = self.lin3(x)

        x = self.dconv1(x)
        x = self.lin4(x)
        x = self.dconv2(x)
        x = self.lin5(x)
        x = self.dconv3(x)

        x += xsave
        x = self.lin6(x)

        return self.out(x)



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_model():
    savemodel = False 
    loadmodel =True 

    if savemodel and loadmodel:
        raise Exception("Can't load model and save it again...")
    

    device = get_device()

    NN = Denoise_Model()
    NN.to(device)

    if loadmodel:
        NN.load_state_dict(torch.load("./models/model.pth", weights_only=True))
        NN.eval()
    else:
        # create training data
        BATCH_SIZE=1
        dataset = myDataset(device, N=10000)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # create some more data for testing
        testdataset = myDataset(device)
        testdataloader = DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(NN.parameters(), lr=1e-4)
        train(NN, dataloader, testdataloader, 10, loss_fn, optimizer, device=device)

        # save model
        if savemodel:
            torch.save(NN.state_dict(), "./models/model.pth")


    
    return NN, device

if __name__ == "__main__":

    NN, device = get_model()

    # try model on example image
    files = glob("./m33/*.fit")

    test_image(NN, files[6], device=device)
#    test_image(NN, "./test_data/m33.fit", device=device)

    plt.show()
