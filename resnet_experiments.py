import fire
import torch
import torchvision
from rich import print

from byol_torch.experiments import LinearExperimentationRegime

def main(encoder='./checkpoints/model+resnet18.pth', epochs=80):
    # Load CIFAR10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    # Test data
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=T.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # get encoder type
    encoder_type = encoder.split('+')[1].split('.')[0]
    if encoder_type == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=False)
    elif encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=False)
    else:
        raise Exception(f"Unknown encoder type found: {encoder_type}")
    
    # Load model
    encoder.load_state_dict(torch.load(encoder))
    
    # begin linear experiment
    experiment = LinearExperimentationRegime(encoder, 1000, 10, train_dataloader, test_dataloader, epochs=epochs)
    res = experiment.run()
    print(f"[green]Linear experiment result: {res}")

if __name__ == "__main__":
    fire.Fire(main)