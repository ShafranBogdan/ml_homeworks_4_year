# train.py
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm, trange

import compute_metrics

def compute_accuracy(preds, targets):
    result = (targets == preds).float().mean()
    return result


def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    return transform


def load_datasets(train_root='CIFAR10/train', test_root='CIFAR10/test', download=False):
    transform = get_transform()
    
    train_dataset = CIFAR10(root=train_root,
                            train=True,
                            transform=transform,
                            download=download)

    test_dataset = CIFAR10(root=test_root,
                           train=False,
                           transform=transform,
                           download=download)
    
    return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size, shuffle=True):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size)
    
    return train_loader, test_loader


def create_model(num_classes=10, zero_init_residual=False, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = resnet18(pretrained=False, num_classes=num_classes, zero_init_residual=zero_init_residual)
    model.to(device)
    
    return model


def train_step(model, images, labels, criterion, optimizer, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    return loss, outputs


def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.inference_mode():
        for test_images, test_labels in test_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            
            outputs = model(test_images)
            preds = torch.argmax(outputs, 1)
            
            all_preds.append(preds)
            all_labels.append(test_labels)
    
    if len(all_preds) > 0 and len(all_labels) > 0:
        accuracy = compute_accuracy(torch.cat(all_preds), torch.cat(all_labels))
        return accuracy
    return torch.tensor(0.0)


def train_model(config, train_loader, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(num_classes=10, 
                        zero_init_residual=config.get("zero_init_residual", False), 
                        device=device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=config["learning_rate"], 
                                 weight_decay=config["weight_decay"])
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for i, (images, labels) in enumerate(train_loader):
            loss, _ = train_step(model, images, labels, criterion, optimizer, device)
            epoch_loss += loss.item()
            num_batches += 1

            if i % 100 == 0:
                accuracy = evaluate_model(model, test_loader, device)
                test_accuracies.append(accuracy.item())
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
    
    final_accuracy = evaluate_model(model, test_loader, device)
    
    return model, final_accuracy, {"train_losses": train_losses, "test_accuracies": test_accuracies}


def save_model(model, path="model.pt"):
    torch.save(model.state_dict(), path)
    return path


def main(config, model_path="model.pt"):
    train_dataset, test_dataset = load_datasets(download=True)
    
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, config["batch_size"], shuffle=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, accuracy, _ = train_model(config, train_loader, test_loader, device)
    
    save_model(model, model_path)
    
    return accuracy


if __name__ == '__main__':
    from hparams import config
    main(config)
