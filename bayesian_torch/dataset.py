import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs):
        'Initialization'
        assert(len(inputs) == len(outputs))
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.inputs[index]
        # no need to load data, as we will control outter
        y = self.outputs[index]

        return x, y

    def register(self, x, y):
        self.inputs.append(x)
        self.outputs.append(y)


class Dataset_T(torch.utils.data.Dataset):
    def __init__(self, inputs, outputs, outputs_t):
        'Initialization'
        assert(len(inputs) == len(outputs))
        self.inputs = inputs
        self.outputs = outputs
        self.outputs_t = outputs_t

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.inputs[index]
        # no need to load data, as we will control outter
        y = self.outputs[index]
        z = self.outputs_t[index]

        return x, y, z

    def register(self, x, y, z):
        self.inputs.append(x)
        self.outputs.append(y)
        self.outputs_t.append(z)

