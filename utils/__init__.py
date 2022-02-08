import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

batch_size = 16

def load_dataset(root='data/'):
    trainset = dsets.MNIST(root= root, 
                              train=True, 
                              transform=transforms.ToTensor(), 
                              download=True)

    testset = dsets.MNIST(root=root,
                             train=False, 
                             transform=transforms.ToTensor(),
                             download=True)


    return trainset, testset

def load_dataloader(trainset, testset, batch_size):

    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)

    testloader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              drop_last=True)
    return trainloader, testloader