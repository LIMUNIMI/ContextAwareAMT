import torch
from torch import nn

from . import settings as s


def conv_output_size(size):
    return int((size - s.DILATION * (s.KERNEL - 1) / s.STRIDE) + 1)


class MIDIParameterEstimation(nn.Module):
    def __init__(self, input_size, output_features):
        """
        Size of the inputs are expected to be 3d: (batch, 1, input_size,
        frames).  Convolutional kernels are applied frame-wise so that the
        input_size is reduced to one and the returned tensor has shape (batch,
        output_features, frames).

        """
        super().__init__()
        # add one block to introduce the needed number of features
        next_input_size = conv_output_size(input_size)
        input_features = 1
        if next_input_size > 0:
            input_size = next_input_size
            self.stack = [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=(s.KERNEL, 1),
                          stride=(s.STRIDE, 1),
                          dilation=(s.DILATION, 1)),
                nn.BatchNorm2d(output_features),
                nn.ReLU()
            ]
            input_features = output_features
        else:
            self.stack = []

        # start adding blocks until we can
        next_input_size = conv_output_size(input_size)
        while next_input_size > 0:
            input_size = next_input_size
            self.stack += [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=(s.KERNEL, 1),
                          stride=(s.STRIDE, 1),
                          dilation=(s.DILATION, 1),
                          groups=input_features),
                nn.BatchNorm2d(output_features),
                nn.ReLU()
            ]
            next_input_size = conv_output_size(input_size)

        # add the last block to get size 1 along frequencies dimension
        if input_size > 1:
            self.stack += [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=(input_size, 1),
                          stride=1,
                          dilation=1,
                          groups=input_features),
                nn.BatchNorm2d(output_features),
                nn.ReLU()
            ]
        self.stack = nn.Sequential(*self.stack)
        print(self)
        print("Total number of parameters: ",
              sum([p.numel() for p in self.parameters() if p.requires_grad]))

    def forward(self, x):
        # we need to remove the height dimension and put features along the
        # channels
        ret = self.stack(x)[:, :, 0]
        # normalize to 1
        # TODO
        return ret / ret.max()


class MIDIVelocityEstimation(MIDIParameterEstimation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        """
        Arguments
        ---------

        inp : torch.tensor
            shape (batch, 1, frames)
        """
        return torch.max(super().forward[:, 0], dim=-1)


def init_weights(m, initializer):
    if hasattr(m, "weight"):
        if m.weight is not None:

            w = m.weight.data
            if w.dim() < 2:
                w = w.unsqueeze(0)
            initializer(w)
