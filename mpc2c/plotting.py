import visdom

import torch

vis = visdom.Visdom()


def plot_losses(trainloss, validloss, trainloss_valid, epoch):
    vis.line(torch.tensor([[trainloss, validloss, trainloss_valid]]),
             X=torch.tensor([epoch]),
             update='append',
             win="losses",
             opts=dict(legend=['train', 'valid', 'trainloss-valid'],
                       title="losses!"))
