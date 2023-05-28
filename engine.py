import fire
import torch
import torchvision
from byol_torch.byol import BYOLNetwork
from byol_torch.train import train

import torchvision.transforms as T

def main(encoder_type='resnet18', epochs=100, device=None, name='model'):
    if encoder_type == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=False)
    elif encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=False)
    else:
        raise Exception(f"Unknown encoder type provided: {encoder_type}")
    
    # Load CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    network = BYOLNetwork(encoder, 1000)
    optimizer = torch.optim.Adam(network.parameters(), lr=3e-4)
    device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    network = network.to(device)
    train(network, optimizer, train_dataloader, device, epochs=epochs)
    
    # save network
    torch.save(network.state_dict(), f"./checkpoints/{name}+{encoder_type}.pth")

    

if __name__ == "__main__":
    fire.Fire(main)