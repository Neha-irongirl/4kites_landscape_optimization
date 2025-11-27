import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from src.models.models import MLP, ResNet20
from src.training.trainer import train
from src.landscape.visualization import (
    create_random_direction,
    get_1d_interpolation,
    get_2d_contour,
    get_weights,
    normalize_direction,
)
from src.landscape.metrics import compute_hessian_eig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')


def main():
    parser = argparse.ArgumentParser(description='Loss Landscape Analysis')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'resnet20'], help='model type')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['synthetic', 'cifar10'], help='dataset name')
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT, help='dataset root for CIFAR-10')
    parser.add_argument('--download_data', action='store_true', help='download CIFAR-10 if missing')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR, help='directory for outputs and checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument('--analysis_samples', type=int, default=512, help='samples reused for landscape analysis')
    parser.add_argument('--hessian_iters', type=int, default=20, help='power-iteration steps for Hessian estimate')
    args = parser.parse_args()

    args.save_dir = os.path.abspath(args.save_dir)
    args.data_root = os.path.abspath(args.data_root)
    os.makedirs(args.save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print('==> Preparing data..')
    trainloader, testloader, trainset = prepare_dataloaders(args)

    print('==> Building model..')
    net = MLP() if args.model == 'mlp' else ResNet20()

    print('==> Training..')
    checkpoint_path = os.path.join(args.save_dir, 'model.pth')
    train(net, trainloader, testloader, epochs=args.epochs, lr=args.lr, save_path=checkpoint_path, device=device)

    if not os.path.exists(checkpoint_path):
        torch.save(net.state_dict(), checkpoint_path)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))

    print('==> Analyzing Landscape..')
    analysis_loader = build_analysis_loader(trainset, args)
    criterion = nn.CrossEntropyLoss()

    print('Computing Hessian Eigenvalues...')
    eig_vals = compute_hessian_eig(
        net,
        analysis_loader,
        criterion,
        device=device,
        top_k=1,
        max_iter=args.hessian_iters,
    )
    print(f'Top Hessian Eigenvalue: {eig_vals[0]:.4f}')
    plot_hessian(eig_vals, os.path.join(args.save_dir, 'hessian.png'))

    print('Generating 1D Interpolation...')
    start_weights = get_weights(net)
    direction = normalize_direction(create_random_direction(net), start_weights, norm='filter')
    end_weights = [w + d for w, d in zip(start_weights, direction)]

    alphas, losses, _ = get_1d_interpolation(
        net,
        analysis_loader,
        start_weights,
        end_weights,
        steps=20,
        device=device,
    )
    plot_1d_curve(alphas, losses, os.path.join(args.save_dir, 'interp_1d.png'))

    print('Generating 2D Contour...')
    x_dir = normalize_direction(create_random_direction(net), start_weights, norm='filter')
    y_dir = normalize_direction(create_random_direction(net), start_weights, norm='filter')
    X, Y, Z = get_2d_contour(
        net,
        analysis_loader,
        start_weights,
        x_dir,
        y_dir,
        steps=10,
        device=device,
    )
    plot_2d_contour(X, Y, Z, os.path.join(args.save_dir, 'contour_2d.png'))

    print('Analysis Complete.')


def prepare_dataloaders(args):
    if args.dataset == 'cifar10':
        os.makedirs(args.data_root, exist_ok=True)
        cifar_folder = os.path.join(args.data_root, 'cifar-10-batches-py')
        expected_files = ['data_batch_1', 'test_batch']
        missing_cache = (
            not os.path.isdir(cifar_folder)
            or any(not os.path.exists(os.path.join(cifar_folder, f)) for f in expected_files)
        )
        download_flag = args.download_data or missing_cache
        if missing_cache and not args.download_data:
            print('Local CIFAR-10 cache missing. Downloading...')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        try:
            trainset = torchvision.datasets.CIFAR10(
                root=args.data_root,
                train=True,
                download=download_flag,
                transform=transform_train,
            )
            testset = torchvision.datasets.CIFAR10(
                root=args.data_root,
                train=False,
                download=download_flag,
                transform=transform_test,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                'CIFAR-10 not found. Supply --download_data or place files under data_root/cifar-10-batches-py.'
            ) from exc
    else:
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = torchvision.datasets.FakeData(
            size=5000,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=base_transform,
            random_offset=0,
        )
        testset = torchvision.datasets.FakeData(
            size=1000,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=base_transform,
            random_offset=1,
        )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = DataLoader(
        testset,
        batch_size=min(256, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return trainloader, testloader, trainset


def build_analysis_loader(trainset, args):
    subset_size = min(len(trainset), args.analysis_samples)
    subset = Subset(trainset, list(range(subset_size)))
    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def plot_hessian(eig_vals, path):
    plt.figure()
    plt.bar([r'$\lambda_1$'], eig_vals)
    plt.ylabel('Value')
    plt.title('Top Hessian Eigenvalue')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_1d_curve(alphas, losses, path):
    plt.figure()
    plt.plot(alphas, losses)
    plt.title('1D Loss Landscape')
    plt.xlabel('Alpha')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_2d_contour(X, Y, Z, path):
    plt.figure()
    contour = plt.contourf(X, Y, Z, levels=20)
    plt.colorbar(contour)
    plt.title('2D Loss Landscape')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    main()