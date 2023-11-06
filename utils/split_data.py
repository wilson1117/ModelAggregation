import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import argparse
import os
from dataset import *

parser = argparse.ArgumentParser(description='Generate dataset for training')
parser.add_argument('--dir', type=str, default='./tmp', help='directory to save files')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset to generate subset')
parser.add_argument('--dataset-cache', default='./datasets', type=str, help='cache path for dataset')
parser.add_argument('--num-client', type=int, default=2, help='number of clients')
parser.add_argument('--fintune-sample', type=int, default=10, help='number of each samples category for finetune')
parser.add_argument('--num-class', type=int, default=10, help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

dataset_dir = os.path.join(args.dir, 'data')
result_img_dir = os.path.join(args.dir, 'imgs')

os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(result_img_dir, exist_ok=True)

dataset, num_classes = load_dataset(args.dataset, dataset_dir, args.dataset_cache, True)

print('loading indices...')
class_index = [[] for _ in range(args.num_class)]
for index, item in tqdm(enumerate(dataset), total=len(dataset)):
    class_index[item[1]] += [index]

distribute = abs(np.random.normal(5, 5, (args.num_client, args.num_class)))
distribute /= distribute.sum(axis=0)

sample_count = np.array([len(item) for item in class_index]) - args.fintune_sample
distribute = (distribute * sample_count).astype(int)
distribute[-1] += sample_count - distribute.sum(axis=0)
distribute = np.vstack((distribute, np.array([args.fintune_sample] * args.num_class)))

bottom_count = np.zeros(args.num_class)

fig, ax = plt.subplots()

for d in distribute:
    ax.bar(range(args.num_class), d, bottom=bottom_count)
    bottom_count += d

plt.savefig(os.path.join(result_img_dir, '{}_split{}_distribute_{}.png'.format(args.dataset, args.num_client, args.seed)))

bottom_idx = np.zeros(args.num_class).astype(int)
output_dataset = []
for dist in distribute:
    indices = []
    for idx, count in enumerate(dist):
        indices.extend(class_index[idx][bottom_idx[idx] : bottom_idx[idx] + count])
        bottom_idx[idx] += count
    
    output_dataset.append(Subset(dataset, indices))

for idx, item in enumerate(output_dataset[:-1]):
    save_dataset(item, os.path.join(dataset_dir, '{}-client{}_{}.pt'.format(args.dataset, idx + 1, args.seed)), num_classes)

save_dataset(output_dataset[-1], os.path.join(dataset_dir, '{}-fintune_{}.pt'.format(args.dataset, args.seed)), num_classes)