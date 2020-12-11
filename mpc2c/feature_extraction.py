from torch import nn


def conv_output_size(size, dilation, kernel, stride):
    return int((size - dilation * (kernel - 1)) / stride)


class MIDIParameterEstimation(nn.Module):
    def __init__(self, input_features, output_features, *hyperparams):
        """
        Size of the inputs are expected to be 3d: (batch, input_features,
        frames).  Convolutional kernels are applied frame-wise so that the
        input_features is reduced to one and the returned tensor has shape
        (batch, output_features, frames), where each output feature corresponds
        to a channel of the output of the stack

        `hyperparams` must contains 3 values:

            * kernel_size
            * stride
            * dilation

        """
        kernel_size, stride, dilation = hyperparams
        super().__init__()
        input_size = input_features
        # add one block to introduce the needed number of features
        next_input_size = conv_output_size(input_size, dilation, kernel_size,
                                           stride)
        input_features = 1
        if next_input_size > 0:
            input_size = next_input_size
            self.stack = [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=(kernel_size, 1),
                          stride=(stride, 1),
                          padding=0,
                          dilation=(dilation, 1)),
                nn.BatchNorm2d(output_features),
                nn.ReLU()
            ]
            input_features = output_features
        else:
            self.stack = []

        # start adding blocks until we can
        next_input_size = conv_output_size(input_size, dilation, kernel_size,
                                           stride)
        while next_input_size > 0:
            input_size = next_input_size
            self.stack += [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=(kernel_size, 1),
                          stride=(stride, 1),
                          dilation=(dilation, 1),
                          padding=0,
                          groups=input_features),
                nn.BatchNorm2d(output_features),
                nn.ReLU()
            ]
            next_input_size = conv_output_size(input_size, dilation,
                                               kernel_size, stride)

        # add the last block to get size 1 along frequencies dimension
        if len(self.stack) == 0:
            raise RuntimeError(
                "Network hyper-parameters would create a one-layer convnet")

        if input_size > 1:
            self.stack += [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=(input_size, 1),
                          stride=1,
                          dilation=1,
                          padding=0,
                          groups=input_features),
                nn.BatchNorm2d(output_features),
                nn.Sigmoid()
            ]
        else:
            # change the last activation so that the outputs fits [0, 1)
            self.stack[-1] = nn.Sigmoid()
        self.stack = nn.Sequential(*self.stack)

    def forward(self, x):
        """
        Arguments
        ---------

        inp : torch.tensor
            shape (batch, in_features, frames)

        Returns
        -------

        torch.tensor
            shape (batch, out_features, frames)
        """
        # unsqueeze the channels dimension
        x = x.unsqueeze(1)
        # apply the stack
        x = self.stack(x)
        # remove the height
        x = x[..., 0, :]
        # !!! output should be included in [0, 1)
        return [
            x,
        ]

    def predict(self, x):
        return self.forward(x)


class MIDIVelocityEstimation(MIDIParameterEstimation):
    def __init__(self, input_features, note_frames, *hyperparams):
        super().__init__(input_features, 1, *hyperparams)
        self.linear = nn.Sequential(nn.Linear(note_frames, note_frames),
                                    nn.ReLU(),
                                    nn.Linear(note_frames, note_frames),
                                    nn.ReLU(), nn.Linear(note_frames, 1),
                                    nn.Sigmoid())

    def forward(self, x):
        """
        Arguments
        ---------

        inp : torch.tensor
            shape (batch, in_features, frames)

        Returns
        -------

        torch.tensor
            shape (batch,)
        """
        return [
            self.linear(super().forward(x)[0][:, 0])[:, 0],
        ]

    def predict(self, x):
        return self.forward(x)


def init_weights(m, initializer):
    if hasattr(m, "weight"):
        if m.weight is not None:

            w = m.weight.data
            if w.dim() < 2:
                w = w.unsqueeze(0)
            initializer(w)
