import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from tqdm import tqdm


class VelocityEstimation(nn.Module):
    def __init__(self,
                 in_numel=MINI_SPEC_SIZE * 100,
                 branches=BRANCHES,
                 k=128 // BRANCHES + 1):
        super().__init__()

        self.preprocess = nn.Sequential(nn.BatchNorm2d(1), nn.Dropout(0.3))

        self.in_numel = in_numel
        self.branches = branches
        self.k = k

        self.process = nn.ModuleList()

        for i in range(branches):
            self.process.append(
                nn.Sequential(nn.Linear(in_numel, k, bias=True), nn.SELU()))

        self.finalize = nn.Sequential(
            nn.Linear(branches * k, branches * k, bias=False), nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False), nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False), nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False), nn.SELU(),
            nn.Linear(branches * k, branches * k, bias=False), nn.SELU(),
            nn.Linear(branches * k, 1, bias=False), nn.Sigmoid())

        # self.apply(lambda x: init_weights(x, nn.init.kaiming_uniform_))

    def forward(self, x):

        # preprocess
        x = self.preprocess(x).reshape(x.shape[0], -1)

        # process each velocity range
        y = torch.zeros(x.shape[0], self.branches,
                        self.k).to(x.dtype).to(x.device)
        for i in range(self.branches):
            y[:, i, :] = self.process[i](x)

        # apply softmax so that only the first output is a probability (classification)
        middle_out = F.softmax(y[:, :, 0], dim=1)

        # finalize takes as input the concatenation of all the features of previous layers
        if self.k > 1:
            x = torch.cat([y[..., i] for i in range(1, self.k)], dim=1)
            x = torch.cat([middle_out, x], dim=1)
        else:
            x = middle_out
        x = self.finalize(x)[:, 0] * 127

        return x, middle_out

    def predict(self, x):
        x = self.forward(x)[0]
        return x
        # return torch.argmax(x, dim=1)


def init_weights(m, initializer):
    if hasattr(m, "weight"):
        if m.weight is not None:

            w = m.weight.data
            if w.dim() < 2:
                w = w.unsqueeze(0)
            initializer(w)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, branches=BRANCHES):
        super().__init__()
        self.inputs = torch.tensor(inputs).to(torch.float).to(DEVICE)
        self.targets = torch.tensor(targets).to(torch.float).to(DEVICE)
        self.targets_middle = torch.zeros(len(targets),
                                          branches).to(torch.float).to(DEVICE)
        self.targets_middle[torch.arange(len(targets)), targets % branches] = 1
        assert len(self.inputs) == len(self.targets),\
            "inputs and targets must have the same length!"
        del inputs, targets

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i], self.targets_middle[i]

    def __len__(self):
        return len(self.inputs)


def train(data, model_path, mini_spec_path):

    print("Loading dataset...")
    mini_spec = open(mini_spec_path, 'rb')
    inputs, targets = pickle.load(mini_spec)
    mini_spec.close()

    print("Building model...")
    model = VelocityEstimation().to(DEVICE)
    print(model)

    # shuffle and split
    indices = list(range(len(inputs) // DATASET_LEN))
    random.seed(1998)
    random.shuffle(indices)
    inputs = np.array(inputs)
    targets = np.array(targets)
    train_size = int(len(indices) * 0.7)
    test_size = valid_size = int(len(indices) * 0.15)
    train_x = inputs[indices[:train_size]]
    valid_x = inputs[indices[train_size:train_size + valid_size]]
    test_x = inputs[indices[-test_size:]]
    train_y = targets[indices[:train_size]]
    valid_y = targets[indices[train_size:train_size + valid_size]]
    test_y = targets[indices[-test_size:]]

    # creating loaders
    trainloader = torch.utils.data.DataLoader(Dataset(train_x, train_y,
                                                      BRANCHES),
                                              batch_size=BATCH_SIZE)
    validloader = torch.utils.data.DataLoader(Dataset(valid_x, valid_y,
                                                      BRANCHES),
                                              batch_size=BATCH_SIZE)
    testloader = torch.utils.data.DataLoader(Dataset(test_x, test_y, BRANCHES),
                                             batch_size=BATCH_SIZE)
    del train_x, train_y, valid_x, valid_y, test_x, test_y, inputs, targets

    optim = torch.optim.Adadelta(model.parameters(), lr=1e-3)

    best_epoch = 0
    best_params = None
    best_loss = 9999
    for epoch in range(EPOCHS):
        print(f"-- Epoch {epoch} --")
        trainloss, validloss = [], []
        print("-> Training")
        model.train()
        for inputs, targets, targets_middle in tqdm(trainloader):
            inputs = inputs.to(DEVICE).unsqueeze(1)
            targets = targets.to(DEVICE)
            targets_middle = targets_middle.to(DEVICE)

            optim.zero_grad()
            out, middle_out = model(inputs)
            bce_loss = F.binary_cross_entropy(middle_out, targets_middle)
            l1_loss = F.l1_loss(out, targets)
            # loss = l1_loss
            loss = bce_loss + l1_loss
            loss.backward()
            optim.step()
            trainloss.append(l1_loss.detach().cpu().numpy())

        print(f"training loss : {np.mean(trainloss)}")

        print("-> Validating")
        with torch.no_grad():
            model.eval()
            for inputs, targets, _ in tqdm(validloader):
                inputs = inputs.unsqueeze(1)
                targets = targets.to(DEVICE)
                # targets = torch.argmax(targets, dim=1).to(torch.float)

                out = model.predict(inputs).to(torch.float)
                loss = torch.abs(targets - out)
                validloss += loss.tolist()

        validloss = np.mean(validloss)
        print(f"validation loss : {validloss}")
        if validloss < best_loss:
            best_loss = validloss
            best_epoch = epoch
            best_params = model.state_dict()
        elif epoch - best_epoch > EARLY_STOP:
            print("-- Early stop! --")
            break

    # saving params
    model.load_state_dict(best_params)
    pickle.dump(model.to('cpu').state_dict(), open(model_path, 'wb'))
    model.to(DEVICE)

    # testing
    print("-> Testing")
    testloss = []
    with torch.no_grad():
        model.eval()
        for inputs, targets, _ in tqdm(testloader):
            inputs = inputs.unsqueeze(1)
            targets = targets.to(DEVICE)
            # targets = torch.argmax(targets, dim=1).to(torch.float)

            out = model.predict(inputs).to(torch.float)
            loss = torch.abs(targets - out)
            testloss += loss.tolist()

        print(
            f"testing absolute error (mean, std): {np.mean(testloss)}, {np.std(testloss)}"
        )
