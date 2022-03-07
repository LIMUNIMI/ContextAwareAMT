import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from . import context


def count_params(model, requires_grad=True):
    """
    Compute the number of parameters
    If `requires_grad` is set to False, parameters that do not require grad
    (freezed) are counted too
    """
    if requires_grad:
        return sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        return sum([p.numel() for p in model.parameters()])


def make_loss_func(loss_func):
    def _loss_fn(x, y, lens):
        x, y, lens = x[0], y[0], lens[0]

        if len(lens) > 1 or lens == torch.tensor(False):
            # if `lens` is False or has multiple values, then it's like note_level
            x = x[..., 0, 0]
            return loss_func(x, y)

        loss = torch.zeros(len(lens))
        for batch, L in enumerate(lens):
            loss[batch] = loss_func(x[batch, :L], y[batch, :L])
        return loss

    return _loss_fn


def best_checkpoint_saver(path):
    @dataclass
    class BestCheckpointSaver(Callback):

        output_path: str

        def on_fit_end(self, trainer, pl_module):
            pl_module.state_dict
            state_dict = pl_module.state_dict()
            fname = Path(self.output_path).with_suffix(".pt")
            torch.save({'state_dict': state_dict}, fname)

    return BestCheckpointSaver(path)
