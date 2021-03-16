import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def conv_output_size(size, dilation, kernel, stride):
    """
    Computes the output size of a convolutional layer.

    N.B. `padding` not supported for now

    Parameters
    ----------

    `size` : tuple-like[int]
        the size of each dimension processed by the convolution (e.g. for 1D
        convolution, use a tuple/list with 1 element; for 2D conv, use a
        tuple/list with 2 elements etc.)

    `dilation`, `kernel`, `stride` : tuple-like[int]
        parameters of the convolution, each for each dimension
    """
    out = []
    for dim in range(len(size)):
        out.append(
            math.floor((size[dim] - dilation[dim] *
                        (kernel[dim] - 1) - 1) / stride[dim] + 1))
    return tuple(out)


class MIDIParameterEstimation(nn.Module):
    def __init__(self, input_size, output_features, note_level, max_layers,
                 dropout, hyperparams):
        """
        * `hyperparams` must contains the following values:

            * kernel_size : tuple[int]
            * stride : tuple[int]
            * dilation : tuple[int]
            * lstm_hidden_size: int [x in 2^x]
            * lstm_layers: int
            * middle_features: int [x in k*(2^x)]
            * middle_activation: int
            * k: int
            * sigmoid_last: tuple

        * `input_size` is a tuple[int] containing the number of rows (features)
        and columns (frames) of each input. It can contain 1 if the size of a
        dimension is unknwon e.g. the number of frames

        * Size of the inputs are expected to be 3d:

            `(batch, input_size[0], input_size[1])`

        * If `note_level` is False, the convolutional kernels are applied
        frame-wise so that the `input_features` dimension is reduced to 1 while
        the `frames` dimension remains untouched.  The returned tensor has
        shape:

            `(batch, output_features, input_size[1])`

        where the `output_feature` dimension is the channel dimension of the
        internal convolutional stack. Only the first int of `input_size` and of
        each hyperparam is used.
        This setting is useful for frame-level parameters.

        * If `note_level` is True, both `hyperparams` and `input_size` should
        contain tuples of 2 ints. In this case, the convolutional stack is made
        so that the output size is:

            `(batch, output_features, 1)`

        This setting can be used for note-level parameters, where each note is
        represented using a fixed number of frames `input_size[1]`. This method
        doesn't work if the number of rows (`input_size[0]`) is much lower than
        the number of columns (`input_size[1]`) [TODO].
        """

        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.note_level = note_level
        self.output_features = output_features

        # setup the `note_level` stuffs
        kernel_size, stride, dilation, lstm_hidden_size,\
            lstm_layers, middle_features, middle_activation,\
            k, sigmoid_last = hyperparams

        middle_features = k * (2**middle_features)

        if lstm_layers > 0:
            conv_in_size = (lstm_hidden_size, input_size[1])
            self.lstm = nn.LSTM(input_size[0],
                                2**lstm_hidden_size,
                                num_layers=lstm_layers,
                                batch_first=True)

        else:
            conv_in_size = input_size

        def add_module(input_features, conv_in_size):
            """
            Add a module to the stack if the output size is >= 0

            Returns (True, size_after_the_module) if the module is added,
            (False, same_size_as_input) if the module is not added.
            """

            # computing size after the first block
            next_conv_in_size = conv_output_size(conv_in_size, dilation,
                                                 kernel_size, stride)

            if next_conv_in_size[0] > 0 and \
                    conv_in_size[0] > 1 and \
                    conv_in_size[0] != next_conv_in_size[0]:
                # if after conv, size is not negative
                # and if the input has something to be reduced
                # and if the conv changes the size (it can happens that
                # dilation creates such a situation)
                if (note_level and next_conv_in_size[1] < 1) or not note_level:
                    # if we cannot apply kernels on the frames, let's
                    # apply them framewise except for the last layer
                    k = (kernel_size[0], 1)
                    s = (stride[0], 1)
                    d = (dilation[0], 1)
                    next_conv_in_size = (next_conv_in_size[0], conv_in_size[1])
                else:
                    k, s, d = kernel_size, stride, dilation

                if next_conv_in_size[0] == 1 and next_conv_in_size[1] == 1:
                    # if this is the last layer
                    conv_out_features = output_features
                else:
                    conv_out_features = middle_features

                self.stack += [
                    nn.Conv2d(input_features,
                              conv_out_features,
                              kernel_size=k,
                              stride=s,
                              padding=0,
                              dilation=d,
                              groups=output_features,
                              bias=False),
                    nn.InstanceNorm2d(conv_out_features,
                                      affine=True,
                                      track_running_stats=True)
                    if conv_out_features > 1 else nn.Identity(),
                    middle_activation()
                ]
                return True, next_conv_in_size
            else:
                return False, conv_in_size

        # start adding blocks until we can
        self.stack = []
        input_features = output_features
        added = True
        for i in range(max_layers):
            added, conv_in_size = add_module(input_features, conv_in_size)
            if not added:
                break
            input_features = middle_features

        # add the last block to get size 1 along feature dimension and
        # (optionally) along the frame dimension
        if len(self.stack) == 0:
            raise RuntimeError(
                "Network hyper-parameters would create a one-layer convnet")

        if not note_level:
            k = (conv_in_size[0], 1)
        else:
            k = conv_in_size

        if k[0] > 1 or k[1] > 1:
            self.stack += [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=k,
                          stride=1,
                          dilation=1,
                          padding=0,
                          groups=output_features,
                          bias=False),
                nn.InstanceNorm2d(
                    output_features, affine=True, track_running_stats=True
                ) if output_features > 1 else nn.Identity()]
            if sigmoid_last:
                self.stack.append(middle_activation())
            else:
                self.stack.append(nn.Sigmoid())

            self.stack.append(
                nn.Conv2d(output_features,
                          output_features,
                          groups=output_features,
                          kernel_size=1))

            if sigmoid_last:
                self.stack.append(nn.Sigmoid())
        else:
            # change the last activation so that the outputs are in (0, 1)
            if not sigmoid_last:
                self.stack[-1] = nn.Sigmoid()
            self.stack.append(
                nn.Conv2d(output_features,
                          output_features,
                          groups=output_features,
                          kernel_size=1))
            if sigmoid_last:
                self.stack.append(nn.Sigmoid())

        self.stack = nn.Sequential(*self.stack)

    def forward(self, x, lens=torch.tensor(False)):
        """
        Arguments
        ---------

        x : torch.tensor
            shape (batch, in_features, frames)

        lens : torch.Tensor
            should contains the lengths of each sample in `x`; do not use if
            only one sample is used or if all the samples have the same length

        Returns
        -------

        torch.tensor
            shape (batch, out_features, frames)
        """
        if hasattr(self, 'lstm'):
            # put the frames before of the features (see nn.LSTM)
            x = torch.transpose(x, 1, 2)
            if lens != torch.tensor(False):
                x = pack_padded_sequence(x, lens, batch_first=True)
            x, _ = self.lstm(x)
            if lens != torch.tensor(False):
                x, lens = pad_packed_sequence(x,
                                              batch_first=True,
                                              padding_value=0)

            # put the frames after the features (see nn.LSTM)
            x = torch.transpose(x, 1, 2)

        x = self.dropout(x)

        # unsqueeze the channels dimension
        x = x.unsqueeze(1).expand(x.shape[0], self.output_features, x.shape[1],
                                  x.shape[2])
        # apply the stack
        x = self.stack(x)
        # remove the height
        x = x[..., 0, :]
        # !!! output should be included in [0, 1)
        return [
            x,
        ]

    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict, start=None, end=None):
        """
        Load parameters contained in `state_dict` starting from `start` and
        ending with `end`, where `start` and `end` are indices of layers of
        the stack of this model, so that the first layer whose parameters are
        loaded is layer with index `start` (included) while the last is `end`
        (not ncluded). The LSTM is always loaded fully.
        """
        # back-up untouched parts (as they are now)
        cp = deepcopy(self.stack)

        # load everything
        super().load_state_dict(state_dict)

        # restore backed-up parts (as they were before)
        stack = list(self.stack)
        if start is not None:
            stack[:start] = cp[:start]
        if end is not None:
            stack[end:] = cp[end:]

        # restore all InstanceNorm layers
        # for i in range(len(stack)):
        #     if type(stack[i]) is nn.InstanceNorm2d:
        #         stack[i] = cp[i]

        self.stack = nn.Sequential(*stack)

    def freeze(self, num_layers=0):
        """
        Set `requires_grad` to `False` for layers until `num_layers`, to `True`
        for the others. Note that normalization layers are not controlled by
        `requires_grad` but by `train` and `eval` mode, so we also set
        `track_running_stats` to False for `InstanceNorm2d` layers before
        `num_layers` and True for the others.

        The lstm is always set to requires_grad=False
        """
        if hasattr(self, 'lstm'):
            self.lstm.requires_grad_(False)
 
        def set_parameters_to(m, boolean):
            for p in m.parameters():
                p.requires_grad = boolean
            if type(m) is nn.InstanceNorm2d:
                m.track_running_stats = boolean

        for m in self.stack[:num_layers]:
            set_parameters_to(m, False)

        for m in self.stack[num_layers:]:
            set_parameters_to(m, True)


class AbsLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class SpaceVariant(nn.Module):
    def __init__(self, module, normalize_func, shape=None):
        """
        A space-variant convolution.
        1. `module` is applied to the input
        2. `normalize_func` is used while creating the map of the positions
        3. indices are extracted from that map
        4. the 2 outputs are multiplied entry-wise

        Arguments
        ---------

        `module` : callable
            e.g. a `torch.nn.Conv2d` object
        `normalize_fun` : callable
            a function which takes as arguments the ndices of one dimension (a
            tensor) and returns another tensor with same dimensions
        `shape` : None or tuple
            the shape expected as input; if you know that, insert it as it
            improves performances
        """
        super().__init__()
        self.module = module
        self.pos = nn.Conv2d(2,
                             1,
                             module.kernel_size,
                             module.stride,
                             module.padding,
                             module.dilation,
                             groups=1,
                             bias=module.bias is not None,
                             padding_mode=module.padding_mode)
        self.normalize = normalize_func
        if shape:
            self.positions = self.make_positions(shape)

    def forward(self, x):
        if hasattr(self, 'positions'):
            positions = self.positions.to(x.dtype).to(x.device)
        else:
            positions = self.make_positions(x.shape).to(x.dtype).to(x.device)
        x = self.module(x)
        positions = self.pos(positions)
        return x * positions

    def make_positions(self, shape):
        """
        Arguments:
        ----------

        `shape` : tuple of int
            the shape of the tensor for which positions should be built.
            4 dimensions: (batches, channels, x, y)
        """
        # making arrays between -1 and +1
        # X = torch.arange(1, shape[-2] + 1)
        # X = self.normalize(X)
        # X = torch.stack([X, torch.flip(X, [0])])
        # # expand X to match Y
        # X = X.unsqueeze(2).expand(2, shape[-2], shape[-1])

        Y = torch.arange(1, shape[-1] + 1)
        Y = self.normalize(Y)
        Y = torch.stack([Y, torch.flip(Y, [0])])
        # expand Y to match X
        Y = Y.unsqueeze(1).expand(2, shape[-2], shape[-1])

        # positions = torch.cat([X, Y], dim=0)
        # positions = torch.stack([X, Y], dim=0)
        # here `positions` has 3 dimensions:
        #   0. dimension 0 has size 2 and contains (X, X_flip, Y, Y_flip)
        #   1. dimension 1 has the indices for x
        #   2. dimension 2 has the indices for y

        # the returned whould have one more position for batches
        return Y.unsqueeze(0).expand(shape[0], 2, shape[2], shape[3])


def init_weights(m, initializer):
    if hasattr(m, "weight"):
        if m.weight is not None:

            w = m.weight.data
            if w.dim() < 2:
                w = w.unsqueeze(0)
            initializer(w)
