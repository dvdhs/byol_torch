import fire
import torch
import torchvision
import os
import numpy as np
from byol_torch.byol import BYOLNetwork
from byol_torch.train import train

import torchvision.transforms as T

def main(encoder_type='resnet18', epochs=100, device=None, name='model', batch_size=256, num_workers=4, dataset='CIFAR10'):
    if encoder_type == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=False)
    elif encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=False)
    else:
        raise Exception(f"Unknown encoder type provided: {encoder_type}")
    
    # Load CIFAR10 dataset
    if dataset == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    elif dataset == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    network = BYOLNetwork(encoder, 1000)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)
    losses = train(network, optimizer, train_dataloader, device, epochs=epochs)
    
    # save network
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./train_logs'):
        os.makedirs('./train_logs')
    np.savetxt(f"./train_logs/{name}+{encoder_type}.csv", losses, delimiter=",")
    torch.save(network.online.encoder.state_dict(), f"./checkpoints/{name}+{encoder_type}.pth")



if __name__ == "__main__":
    fire.Fire(main)