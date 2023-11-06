import torch
from torchvision import transforms, datasets

def load_dataset(dataset, cache_dir, train, normalize=True):
    if hasattr(datasets, dataset):
        if dataset == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
            ]) if normalize else transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset = datasets.CIFAR10(root=cache_dir, train=train, download=True, transform=transform)
        
        return dataset, len(dataset.classes)
    
    return torch.load(dataset)