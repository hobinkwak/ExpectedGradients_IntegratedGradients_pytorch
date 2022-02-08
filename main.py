import numpy as np
import torch
from model import CNN
from utils import load_dataset
from attr import integrated_gradients, expected_gradients

if __name__ == '__main__':

    trainset, testset = load_dataset()
    model = torch.load('model/cnn.pt').to('cuda')

    # Target Number : 0 ~ 9
    idx = 42
    target = 3
    n_iter = 1000
    baseline = testset.test_data.view(len(testset), 1, 28,28).float().numpy()
    data = baseline[np.where(testset.test_labels == target)][idx:idx+1]

    ig_attr = integrated_gradients(model,
                                 data,
                                 target,
                                 n_iter,
                                 device='cuda',
                                 visualize=True)
    eg_attr = expected_gradients(model,
                                 data,
                                 baseline,
                                 target,
                                 n_iter,
                                 device='cuda',
                                 visualize=True)
