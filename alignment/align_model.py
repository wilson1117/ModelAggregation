import path
import sys
path = path.Path().parent.abspath()
sys.path.append(path)

import argparse
import torch
from torch.utils.data import DataLoader
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='./tmp', help='directory to save files')
    parser.add_argument('--models', type=str, nargs=2, help='models to align')
    parser.add_argument('--align-dataset', type=str, required=True, help='dataset to align')
    parser.add_argument('--dataset-cache', default='./datasets', type=str, help='cache path for dataset')

    args = parser.parse_args()

    model_dir = os.path.join(args.dir, 'models')
    dataset_dir = os.path.join(args.dir, 'data')

    data, _ = load_dataset(args.align_dataset, dataset_dir, args.dataset_cache, False)
    loader = DataLoader(data, batch_size=len(data), shuffle=False, num_workers=6)

    models = [load_model(os.path.join(model_dir, model_path))[0] for model_path in args.models]

    if len(models) < 2:
        raise ValueError('At least 2 models are required')

    feats = [next(iter(loader))[0]] * len(models)
    for layer in range(len(models[0].layers)):
        y = [models[idx][layer](feats[idx]) for idx in range(len(models))]
        print(y)
        break