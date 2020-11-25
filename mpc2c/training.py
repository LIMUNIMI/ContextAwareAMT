from time import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from . import data_management, feature_extraction, plotting
from . import settings as s


def train_pedaling(nmf_params):
    trainloader = data_management.get_loader(['train'], nmf_params, 'pedaling')
    validloader = data_management.get_loader(['valid'], nmf_params, 'pedaling')
    model = feature_extraction.MIDIParameterEstimation(s.BINS, 3).to(s.DEVICE)
    train(trainloader, validloader, model)


def train_velocity(nmf_params):
    trainloader = data_management.get_loader(['train'], nmf_params, 'velocity')
    validloader = data_management.get_loader(['valid'], nmf_params, 'velocity')
    model = feature_extraction.MIDIVelocityEstimation(s.BINS, 1).to(s.DEVICE)

    train(trainloader, validloader, model)


def train(trainloader, validloader, model):
    optim = torch.optim.Adadelta(model.parameters(), lr=s.LR_VELOCITY)

    def trainloss_fn(x, y):
        return F.l1_loss(x * 127, y)

    validloss_fn = trainloss_fn
    train_epoch(model, optim, trainloss_fn, validloss_fn, trainloader,
                validloader)


def train_epoch(model, optim, trainloss_fn, validloss_fn, trainloader,
                validloader):
    """
    A typical training algorithm with early stopping and loss plotting
    """
    best_epoch = 0
    best_loss = 9999
    for epoch in range(s.EPOCHS):
        epoch_ttt = time()
        print(f"-- Epoch {epoch} --")
        trainloss, validloss, trainloss_valid = [], [], []
        print("-> Training")
        model.train()
        for inputs, targets in tqdm(trainloader):
            inputs = inputs.to(s.DEVICE)
            targets = targets.to(s.DEVICE)

            optim.zero_grad()
            out = model(inputs)
            loss = trainloss_fn(out, targets)
            loss.backward()
            optim.step()
            trainloss.append(loss.detach().cpu().numpy())

        print(f"training loss : {np.mean(trainloss):.4e}")

        print("-> Validating")
        with torch.no_grad():
            model.eval()
            for inputs, targets in tqdm(validloader):
                inputs = inputs
                targets = targets.to(s.DEVICE)
                # targets = torch.argmax(targets, dim=1).to(torch.float)

                out = model(inputs)
                loss = validloss_fn(targets, out)
                validloss += loss.tolist()
                trainloss_valid += trainloss_fn(targets, out)

        validloss = np.mean(validloss)
        trainloss_valid = np.mean(validloss_fn)
        print(f"validation loss : {validloss:.4e}")
        print(f"validation-training loss : {validloss:.4e}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            state_dict = model.cpu().state_dict()
            name = f"checkpoints/checkpoint{best_loss:.4f}.pt"
            torch.save({'dtype': model.dtype, 'state_dict': state_dict}, name)
        elif epoch - best_epoch > s.EARLY_STOP:
            print("-- Early stop! --")
            break
        else:
            print(f"{epoch - best_epoch} from early stop!!")

        if s.PLOT_LOSSES:
            plotting.plot_losses(trainloss, validloss, trainloss_valid, epoch)

        print("Time for this epoch: ", time() - epoch_ttt)
