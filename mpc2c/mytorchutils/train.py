import visdom
from time import time

import numpy as np
import torch
from tqdm import tqdm


def train_epochs(model,
                 optim,
                 trainloss_fn,
                 validloss_fn,
                 trainloader,
                 validloader,
                 epochs: int = 500,
                 early_stop: int = 15,
                 plot_losses: bool = True,
                 device: str = 'cuda',
                 dtype=torch.float32):
    """
    A typical training algorithm with early stopping and loss plotting

    Arguments
    ---------
    `model` : torch.nn.Model

    `optim` : torch.optim.Optimizer

    `trainloss_fn` : callable
        a loss function which accepts 3 arguments:

        * targets
        * predictions
        * length of the inputs along the batch dimension

    `validloss_fn` : callable
        a loss function which accepts 3 arguments:

        * targets
        * predictions
        * length of the inputs along the batch dimension

    `trainloader` : torch.utils.data.DataLoader
        must yield:

        * `inputs` : list
            a list of tensors that will be used as inputs to the model (nn
            that need multiple inputs can still use this!): each input will be
            an argument to the model forward pass
        * `targets` : list
            a list of tensors that will be passed to the loss functions as it
            is (nn that have intermediate targets can still use that!)
        * `lens` : list
            a list of tensors where each element contains the length of the
            corresponding target sample

    `validloader` : torch.utils.data.DataLoader
        same as `trainloader` but used for validation

    `epochs` : int
        the maximum number of epochs (defaults to 500)

    `early_stop` : int
        how many consecutive epochs with valid loss greater of the lowest valid
        loss are allowed before of stopping the training (defaults to 15)

    `plot_losses` : bool
        if True, use `visdom` to plot losses while training (defaults to True)

    `device` : str
        the device that will be used, e,g. `cuda`, `cpu`, `cuda:0` `cuda:1`
        (defaults to 'cuda')

    `dtype` : torch.dtype
        the dtype to use for the data (defaults to torch.float32)

    Note
    ----
    This algorithms correctly handles cases in which each sample has a
    different length.

    Moreover, it allows to use two different losses for training and
    validation. The method will use the training loss also during validation
    for comparison purposes; however, sometimes you are interested in metric
    different from the loss.

    You should thus use the best loss according to the backpropagation
    algorithm as training loss and your target metric as validation loss; this
    approach allows you to monitor the metric of interest during validation, to
    use early stopping against your metric of interest and to monitor the
    validation comparing it with the training.
    """
    best_epoch = 0
    best_loss = 9999
    for epoch in range(epochs):
        epoch_ttt = time()
        print(f"-- Epoch {epoch} --")
        trainloss, validloss, trainloss_valid = [], [], []
        print("-> Training")
        model.train()
        for inputs, targets, lens in tqdm(trainloader):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].to(device).to(dtype)
            for i in range(len(targets)):
                targets[i] = targets[i].to(device).to(dtype)

            optim.zero_grad()
            out = model(inputs)
            loss = trainloss_fn(out, targets, lens)
            loss.backward()
            optim.step()
            trainloss.append(loss.detach().cpu().numpy())

        trainloss = np.mean(trainloss)
        print(f"training loss : {trainloss:.4e}")

        print("-> Validating")
        with torch.no_grad():
            model.eval()
            for inputs, targets, lens in tqdm(validloader):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(device).to(dtype)
                for i in range(len(targets)):
                    targets[i] = targets[i].to(device).to(dtype)

                out = model(*inputs)
                validloss.append(
                    validloss_fn(targets, out, lens).detach().cpu().numpy())
                trainloss_valid.append(
                    trainloss_fn(targets, out, lens).detach().cpu().numpy())

        validloss = np.mean(validloss)
        trainloss_valid = np.mean(trainloss_valid)
        print(f"validation loss : {validloss:.4e}")
        print(f"validation-training loss : {validloss:.4e}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            state_dict = model.cpu().state_dict()
            name = f"checkpoints/checkpoint{best_loss:.4f}.pt"
            torch.save({'dtype': model.dtype, 'state_dict': state_dict}, name)
        elif epoch - best_epoch > early_stop:
            print("-- Early stop! --")
            break
        else:
            print(f"{epoch - best_epoch} from early stop!!")

        if plot_losses:
            plot_losses(trainloss, validloss, trainloss_valid, epoch)

        print("Time for this epoch: ", time() - epoch_ttt)
    return best_loss


vis = visdom.Visdom()


def plot_losses(trainloss, validloss, trainloss_valid, epoch):
    vis.line(torch.tensor([[trainloss, validloss, trainloss_valid]]),
             X=torch.tensor([epoch]),
             update='append',
             win="losses",
             opts=dict(legend=['train', 'valid', 'trainloss-valid'],
                       title="losses!"))
