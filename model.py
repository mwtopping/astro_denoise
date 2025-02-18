import torch.nn as nn
import cv2 as cv
from astropy.visualization import ZScaleInterval
from astropy.io import fits
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch


# TODO: create two images; one with narrower stars than the other
#  Save gaussian widths and reacreate at 80% or something
# - allow for different x and y sigma?
def make_img(Npoints, size):
    img = np.zeros((size, size))
    fine_img = np.zeros((size, size))
    for ii in range(Npoints):
        s = np.random.uniform(0.001, 0.05)
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        img += gauss2d(x, 
                       y,
                      s,s,size)
        fine_img += gauss2d(x,y,0.85*s,0.85*s,size)


    return img, fine_img

def gauss2d(mux, muy, sigmax, sigmay, size):
    xx = np.linspace(0, 1, size)
    yy = np.linspace(0, 1, size)
    XX, YY = np.meshgrid(xx, yy)

    img = 1. / (2. * np.pi * sigmax * sigmay) * np.exp(-((XX - mux)**2. / (2. * sigmax**2.) + (YY - muy)**2. / (2. * sigmay**2.)))

    img = img - np.nanmedian(img)
#    img /= np.nanstd(img)


    return np.array(img)



def add_noise(img, fine_img, amount=1.0):
    img = img - np.nanmean(img)
    fine_img = fine_img - np.nanmean(fine_img)
    if np.nanstd(img) != 0:
        img /= np.nanstd(img)
        fine_img /= np.nanstd(fine_img)

    nimg = img + np.random.normal(loc=0, scale=amount, size=img.shape)

    return fine_img, nimg

class Denoiser(nn.Module):
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
        x = x[:,None,:,:]
#        exit()
        x = self.conv1(x)
        x = self.lin1(x)
        xsave = x
        x = self.conv2(x)
        x = self.lin2(x)
        prepool_shape = x.shape
#        x = self.pool(x)
        x = self.dconv1(x)
#        x = self.unpool(x, idxs, output_size=prepool_shape)
        x = self.lin2(x)
        x = self.dconv2(x)

        x += xsave
#        x = self.unpool(x, idxs, output_size=prepool_shape)
        x = self.lin3(x)

        return self.out(x)


class myDataset(Dataset):
    def __init__(self, device, transform=None):
        super().__init__()
        self.images = []
        self.targets = []
        self.device = device

        N = 5000
        for ii in range(N):
            img, fine_img = make_img(np.random.randint(0, 8), 64)

            img, nimg = add_noise(img, fine_img, amount=np.random.uniform(1, 3))

            self.targets.append(img)
            self.images.append(nimg)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return torch.tensor(self.images[idx], dtype=torch.float).to(self.device), torch.tensor(self.targets[idx], dtype=torch.float).to(self.device)

def calc_loss(inp_batch, targ_batch, model, device):
    inp_batch.to(device)
    targ_batch.to(device)


def evalutae_model(model, train_loader, test_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, test_loss



def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.mse_loss(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss

    return total_loss / num_batches



def train(model, dataloader, testdataloader, Nepochs, loss_fn, device="cpu"):
    train_losses = []
    test_losses = []
    steps = []
    global_step = -1
    eval_freq = 50
    for ii in tqdm(range(Nepochs)):
        total_loss = 0
        model.train()
        for inp_batch, targ_batch in dataloader:
            optimizer.zero_grad()

            # calc the loss
            output = model(inp_batch)
            loss = loss_fn(output, targ_batch)
            loss.backward()
            optimizer.step()
            if global_step % eval_freq == 0:
                train_loss, test_loss = evalutae_model(model, dataloader, testdataloader, device, 30)
    
                train_losses.append(train_loss.cpu())
                test_losses.append(test_loss.cpu())
                steps.append(global_step)

            global_step += 1

    plt.figure()
    plt.plot(steps, train_losses)
    plt.plot(steps, test_losses)
    plt.xscale('log')
    plt.yscale('log')


def process_full_image(model, image, size, device):
    image_shape = np.shape(image)

    new_img = np.zeros_like(image).astype(np.float32)
    mask = np.zeros_like(image).astype(np.float32)

    ii = 0

    for ix in range(0, image_shape[0]-size, int(size/2)):
        for iy in range(0, image_shape[1]-size, int(size/2)):
            print(f"\rProcessing tile number {ii}", end="")

            crop = image[ix:ix+size, iy:iy+size].astype(np.float64)

            crop -= np.nanmean(image[ix:ix+size, iy:iy+size])
            crop /= np.nanstd(image[ix:ix+size, iy:iy+size])

            cropmedian = np.nanmedian(crop)
            test = NN(torch.tensor(crop, dtype=torch.float)[None, :, :].to(device))

            out = test[0][0].cpu().detach().numpy().astype(np.float32)
            outmedian = np.nanmedian(out)

            out = out-outmedian+cropmedian

            out *= np.nanstd(image[ix:ix+size, iy:iy+size])
            out += np.nanmean(image[ix:ix+size, iy:iy+size])

            pad = 2

            new_img[ix+pad:ix+size-pad, iy+pad:iy+size-pad] += out[pad:-pad, pad:-pad]
            mask[ix+pad:ix+size-pad, iy+pad:iy+size-pad] += np.ones_like(out[pad:-pad, pad:-pad])
            ii += 1

    new_img /= mask
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    limits = ZScaleInterval().get_limits(image)
    ax[0][0].imshow(image, vmin=limits[0], vmax=limits[1], cmap='Greys_r')
    ax[0][1].imshow(new_img, vmin=limits[0], vmax=limits[1], cmap='Greys_r')
    ax[1][0].imshow((0.3*image+0.7*new_img), vmin=limits[0], vmax=limits[1], cmap='Greys_r')
    ax[1][1].imshow(cv.GaussianBlur(new_img, (5, 5),0), vmin=limits[0], vmax=limits[1], cmap='Greys_r')

if __name__ == "__main__":


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.device(device)



    BATCH_SIZE=1
    dataset = myDataset(device)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    testdataset = myDataset(device)
    testdataloader = DataLoader(testdataset, batch_size=1, shuffle=True, num_workers=0)
    # test some random image creation


    # Download the FashionMNIST dataset
#    trainset = torchvision.datasets.FashionMNIST(root='./data', train = True, download= True, transform=transform)
#    testset = torchvision.datasets.FashionMNIST(root='./data', train = False, download= True, transform=transform)

    # Load the dataset in the dataloader
#    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
#    testLoader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)


    NN = Denoiser()
    NN.to(device)

#    print(NN.parameters.is_cuda)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(NN.parameters(), lr=1e-4)
    train(NN, dataloader, testdataloader, 10, loss_fn, device=device)
    NN.eval()
    inp, tar = dataset[0]
    fig, axs = plt.subplots(4, 3)
    for ii in range(3):
        img, fine_img = make_img(np.random.randint(0, 8), 64)
        img, nimg = add_noise(img, fine_img, amount=1)

        axs[ii][0].imshow(img)
        axs[ii][1].imshow(nimg)
        test = NN(torch.tensor(img, dtype=torch.float)[None, :, :].to(device))
        axs[ii][2].imshow(test[0][0].cpu().detach().numpy())
    axs[0][0].set_title("Original Image")
    axs[0][1].set_title("Noisy Image")
    axs[0][2].set_title("Processed Image")

    data = fits.getdata('./data/m33.fit')
    corner = [1224, 2230]
    size = 64
    crop = data[corner[0]:corner[0]+size, corner[1]:corner[1]+size].astype(np.float64)
    crop -= np.nanmean(crop)
    crop /= np.nanstd(crop)

    process_full_image(NN, data, size, device)
    plt.show()
