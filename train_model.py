import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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



def train(model, dataloader, testdataloader, Nepochs, loss_fn, optimizer, device=torch.device("cpu")):
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


