import torch
from torchvision import transforms, datasets
import os

def load_dataset(dataset, dataset_dir, cache_dir, train, normalize=True):
    if hasattr(datasets, dataset):
        cache_dir = os.path.join(cache_dir, dataset)
        if dataset == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]),
            ]) if normalize else transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset = datasets.CIFAR10(root=cache_dir, train=train, download=True, transform=transform)
        
        return dataset, len(dataset.classes)
    
    return torch.load(os.path.join(dataset_dir, dataset))

def save_dataset(dataset, save_path, num_classes):
    torch.save([dataset, num_classes], save_path)
    print(f"Dataset saved to {save_path}")