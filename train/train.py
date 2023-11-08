import path
import sys
path = path.Path().parent.abspath()
sys.path.append(path)

import argparse
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
import models
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str, default='./tmp', help='directory to save files')
    parser.add_argument('--model', choices=[ model for model in dir(models) if model[0] != '_'], default='VGG16')
    parser.add_argument('--trainset', required=True)
    parser.add_argument('--train_batch', default=64, type=int)
    parser.add_argument('--testset', required=True)
    parser.add_argument('--test_batch', default=512, type=int)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--eval-epoch', default=5, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset-cache', default='./datasets', type=str, help='cache path for dataset')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset_dir = os.path.join(args.dir, 'data')
    result_img_dir = os.path.join(args.dir, 'imgs')
    model_dir = os.path.join(args.dir, 'models')

    os.makedirs(model_dir, exist_ok=True)

    trainset_name = args.trainset.split(os.path.sep)[-1].split('.')[0]

    trainset, num_classes = load_dataset(args.trainset, dataset_dir, args.dataset_cache, True)
    testset, num_classes = load_dataset(args.testset, dataset_dir, args.dataset_cache, False)

    trainloader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=4)
    testloader = DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = getattr(models, args.model)
    model = arch(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)

    best_acc = 0.0
    print("Start training...")
    for epoch in range(args.epoch):
        model.train()

        train_loss = 0.0
        correct = 0
        
        for feat, label in tqdm(trainloader):
            optimizer.zero_grad()
            feat, label = feat.to(device), label.to(device)
            pred = model(feat)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * feat.size(0)
            correct += (pred.argmax(dim=1) == label).sum().item()
        
        train_loss /= len(trainset)
        acc = correct / len(trainset) * 100
        print(f"Epoch {epoch + 1} / {args.epoch}, train loss: {train_loss :.3f}, train acc: {acc :.2f}")

        if (epoch + 1) % args.eval_epoch == 0:
            model.eval()
            test_loss = 0.0
            correct = 0

            with torch.no_grad():
                for feat, label in tqdm(testloader):
                    feat, label = feat.to(device), label.to(device)
                    pred = model(feat)
                    loss = criterion(pred, label)

                    test_loss += loss.item() * feat.size(0)
                    correct += (pred.argmax(dim=1) == label).sum().item()
            
            test_loss /= len(testset)
            acc = correct / len(testset) * 100
            print(f"Epoch {epoch + 1} / {args.epoch}, test loss: {test_loss :.3f}, test acc: {acc :.2f}")

            if acc > best_acc:
                best_acc = acc
                save_model(
                    arch,
                    model.state_dict(),
                    os.path.join(model_dir, f"{args.model}-{trainset_name}_{args.seed}_best.pt"),
                    seed = args.seed,
                    epoch = epoch,
                    acc = acc,
                    trainset = args.trainset
                )

    save_model(
        arch,
        model.state_dict(),
        os.path.join(model_dir, f"{args.model}-{trainset_name}_{args.seed}.pt"),
        seed = args.seed,
        epoch = epoch,
        acc = acc,
        trainset = args.trainset
    )