import fire
import torch
import torchvision
import torchvision.transforms as T
from rich import print

from byol_torch.experiments import LinearExperimentationRegime

def main(encoder_path='./checkpoints/model+resnet18.pth', epochs=80, dataset='CIFAR10'):
    # Load the datasets
    if dataset == 'CIFAR10':
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

        # Test data
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=T.ToTensor())
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == 'CIFAR100':
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

        # Test data
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=T.ToTensor())
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    elif dataset == 'STL10':
        train_dataset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=T.ToTensor())
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

        # Test data
        test_dataset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=T.ToTensor())
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    else:
        raise Exception(f"Unknown dataset provided: {dataset}")
    num_classes = 10 if dataset == 'CIFAR10' else 100
    # get encoder type
    encoder_type = encoder_path.split('+')[1].split('.')[0]
    if encoder_type == 'resnet18':
        encoder = torchvision.models.resnet18(pretrained=False)
    elif encoder_type == 'resnet50':
        encoder = torchvision.models.resnet50(pretrained=False)
    else:
        raise Exception(f"Unknown encoder type found: {encoder_type}")
    
    # Load model
    encoder.load_state_dict(torch.load(encoder_path))
    
    # begin linear experiment
    experiment = LinearExperimentationRegime(encoder, 1000, 100, train_dataloader, test_dataloader, epochs=epochs)
    res = experiment.run()
    print(f"[green]Linear experiment result: {res}")

if __name__ == "__main__":
    fire.Fire(main)