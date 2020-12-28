from torch import nn


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
            int((size[dim] - dilation[dim] *
                 (kernel[dim] - 1)) / stride[dim]) + 1)
    return tuple(out)


class MIDIParameterEstimation(nn.Module):
    def __init__(self, input_size, output_features, note_level, hyperparams):
        """
        * `hyperparams` must contains 3 values:

            * kernel_size : tuple[int]
            * stride : tuple[int]
            * dilation : tuple[int]

        * `input_size` is a tuple[int] containing the number of rows (features)
        and columns (frames) of each input. It can contain only one number if
        the number of columns is unknwon.

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

        # setup the `note_level` stuffs
        kernel_size, stride, dilation = hyperparams

        def add_module(input_features):
            """
            Add a module to the stack if the output size is >= 0

            Returns (True, size_after_the_module) if the module is added,
            (False, same_size_as_input) if the module is not added.
            """

            # computing size after the first block
            next_input_size = conv_output_size(input_size, dilation,
                                               kernel_size, stride)

            if next_input_size[0] > 0 and \
                    input_size[0] > 1 and \
                    input_size[0] != next_input_size[0]:
                # if after conv, size is not negative
                # and if the input has something to be reduced
                # and if the conv changes the size (it can happens that
                # dilation creates such a situation)
                if (note_level and next_input_size[1] < 1) or not note_level:
                    # if we cannot apply kernels on the frames, let's
                    # apply them framewise except for the last layer
                    k = (kernel_size[0], 1)
                    s = (stride[0], 1)
                    d = (dilation[0], 1)
                    next_input_size = (next_input_size[0], input_size[1])
                else:
                    k, s, d = kernel_size, stride, dilation

                self.stack += [
                    nn.Conv2d(input_features,
                              output_features,
                              kernel_size=k,
                              stride=s,
                              padding=0,
                              dilation=d),
                    nn.BatchNorm2d(output_features),
                    nn.ReLU()
                ]
                return True, next_input_size
            else:
                return False, input_size

        # start adding blocks until we can
        self.stack = []
        input_features = 1
        added = True
        while added:
            added, input_size = add_module(input_features)
            input_features = output_features

        # add the last block to get size 1 along feature dimension and
        # (optionally) along the frame dimension
        if len(self.stack) == 0:
            raise RuntimeError(
                "Network hyper-parameters would create a one-layer convnet")

        if not note_level:
            k = (input_size[0], 1)
        else:
            k = input_size

        if k[0] > 1 or k[1] > 1:
            self.stack += [
                nn.Conv2d(input_features,
                          output_features,
                          kernel_size=k,
                          stride=1,
                          dilation=1,
                          padding=0,
                          groups=input_features),
                # nn.BatchNorm2d(output_features),
                nn.Sigmoid()
            ]
        else:
            # remove the batchnorm since the output has only one value
            # and it cannot be run with these input size
            del self.stack[-2]
            # change the last activation so that the outputs are in (0, 1)
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


# class MIDIVelocityEstimation(MIDIParameterEstimation):
#     def __init__(self, input_features, note_frames, hyperparams):
#         super().__init__(input_features, 1, hyperparams)
#         self.linear = nn.Sequential(nn.Linear(note_frames, note_frames),
#                                     nn.ReLU(),
#                                     nn.Linear(note_frames, note_frames),
#                                     nn.ReLU(), nn.Linear(note_frames, 1),
#                                     nn.Sigmoid())

#     def forward(self, x):
#         """
#         Arguments
#         ---------

#         inp : torch.tensor
#             shape (batch, in_features, frames)

#         Returns
#         -------

#         torch.tensor
#             shape (batch,)
#         """
#         return [
#             self.linear(super().forward(x)[0][:, 0])[:, 0],
#         ]

#     def predict(self, x):
#         return self.forward(x)


def init_weights(m, initializer):
    if hasattr(m, "weight"):
        if m.weight is not None:

            w = m.weight.data
            if w.dim() < 2:
                w = w.unsqueeze(0)
            initializer(w)
