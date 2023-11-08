import path
import sys
path = path.Path().parent.abspath()
sys.path.append(path)

from models import *
import argparse
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ConfusionMatrixDisplay
from utils import *

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./tmp', help='directory to save files')
    parser.add_argument('--model', required=True)
    parser.add_argument('--testset', required=True)
    parser.add_argument('--test_batch', default=512, type=int)
    parser.add_argument('--dataset-cache', default='./datasets', type=str, help='cache path for dataset')

    args = parser.parse_args()

    dataset_dir = os.path.join(args.dir, 'data')
    model_dir = os.path.join(args.dir, 'models')
    result_img_dir = os.path.join(args.dir, 'imgs')

    os.makedirs(result_img_dir, exist_ok=True)

    testset, num_classes = load_dataset(args.testset, dataset_dir, args.dataset_cache, False)
    testloader = DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    model, _ = load_model(os.path.join(model_dir, args.model))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()

    cm = np.zeros((num_classes, num_classes))
    correct = 0
    for images, labels in tqdm(testloader):
        images, labels = images.to(device), labels.to(device)
        predicted = model(images.to(device)).argmax(dim=1)

        for i in range(len(predicted)):
            cm[predicted[i]][labels[i]] += 1

        correct += (predicted == labels).sum().item()

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.title('Accuracy: %.2f%%' % (100 * correct / len(testset)))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, "{:.2f}".format(cm[i, j] / cm[: ,j].sum()),
                horizontalalignment="center", color=("black" if cm[i, j] / cm[: ,j].sum() < 0.5 else 'white'))

    model_name = '.'.join(args.model.split('.')[:-1])
    testset_name = args.testset.split(os.path.sep)[-1].split('.')[0]

    print(f"Accuracy: {100 * correct / len(testset)}%")

    plt.savefig(os.path.join(result_img_dir, f"{model_name}_{testset_name}.png"))