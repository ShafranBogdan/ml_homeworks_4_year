import os
import pytest
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
import json
import sys

from hparams import config
from train import (
    compute_accuracy, 
    get_transform, 
    load_datasets, 
    create_data_loaders, 
    create_model, 
    train_step,
    evaluate_model, 
    train_model,
    save_model
)


@pytest.fixture
def transform():
    return get_transform()


@pytest.fixture
def temp_dir():
    return tempfile.gettempdir()


@pytest.fixture
def datasets(temp_dir):
    train_root = os.path.join(temp_dir, 'CIFAR10/train')
    test_root = os.path.join(temp_dir, 'CIFAR10/test')
    return load_datasets(train_root, test_root, download=True)


@pytest.fixture
def data_loaders(datasets):
    train_dataset, test_dataset = datasets
    return create_data_loaders(train_dataset, test_dataset, batch_size=32)


def test_compute_accuracy():
    preds = torch.tensor([1, 2, 3, 4, 5])
    targets = torch.tensor([1, 2, 3, 0, 0])
    accuracy = compute_accuracy(preds, targets)
    assert accuracy == 0.6
    
    preds = torch.tensor([1, 2, 3, 4, 5])
    targets = torch.tensor([1, 2, 3, 4, 5])
    accuracy = compute_accuracy(preds, targets)
    assert accuracy == 1.0


def test_get_transform():
    transform = get_transform()
    
    assert isinstance(transform, transforms.Compose)

    assert any(isinstance(t, transforms.ToTensor) for t in transform.transforms)
    assert any(isinstance(t, transforms.Normalize) for t in transform.transforms)


def test_load_datasets(temp_dir):
    train_root = os.path.join(temp_dir, 'CIFAR10/train')
    test_root = os.path.join(temp_dir, 'CIFAR10/test')
    
    train_dataset, test_dataset = load_datasets(train_root, test_root, download=True)
    
    assert isinstance(train_dataset, CIFAR10)
    assert isinstance(test_dataset, CIFAR10)
    assert len(train_dataset) > 0
    assert len(test_dataset) > 0
    assert train_dataset.train
    assert not test_dataset.train


def test_create_data_loaders(datasets):
    train_dataset, test_dataset = datasets
    
    train_loader, test_loader = create_data_loaders(train_dataset, test_dataset, batch_size=16)
    
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert train_loader.batch_size == 16
    assert test_loader.batch_size == 16
    
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    
    assert train_images.shape[0] <= 16  # Размер батча не больше batch_size
    assert test_images.shape[0] <= 16
    assert train_images.shape[1:] == (3, 32, 32)  # Формат изображений CIFAR10


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_create_model(device):
    device = torch.device(device)
    
    # Тест с указанным устройством
    model = create_model(device=device)
    assert next(model.parameters()).device.type == device.type
    
    # Тест с различными параметрами
    model_custom = create_model(num_classes=20, zero_init_residual=True, device=device)
    assert model_custom.fc.out_features == 20


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_train_step(device, datasets):
    device = torch.device(device)
    train_dataset, _ = datasets
    
    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    images, labels = next(iter(train_loader))
    
    loss, outputs = train_step(model, images, labels, criterion, optimizer, device)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert torch.isfinite(loss)
    assert outputs.shape[0] == images.shape[0]  # Проверяем, что выходы модели имеют правильную форму
    assert outputs.shape[1] == 10  # 10 классов для CIFAR10


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_evaluate_model(device, data_loaders):
    device = torch.device(device)
    _, test_loader = data_loaders
    
    model = create_model(device=device)
    
    accuracy = evaluate_model(model, test_loader, device)
    
    assert isinstance(accuracy, torch.Tensor)
    assert 0 <= accuracy <= 1


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_save_model(device, temp_dir):
    device = torch.device(device)
    
    model = create_model(device=device)
    
    model_path = os.path.join(temp_dir, "test_model.pt")
    saved_path = save_model(model, model_path)
    
    assert os.path.exists(saved_path)
    assert saved_path == model_path
    
    loaded_state_dict = torch.load(model_path)
    assert isinstance(loaded_state_dict, dict)
    
    new_model = create_model(device=device)
    new_model.load_state_dict(loaded_state_dict)


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_train_on_one_batch(device, datasets):
    """Тест обучения на одном батче."""
    device = torch.device(device)
    train_dataset, _ = datasets
    batch_size = 16
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(train_loader))
    
    model = create_model(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    loss, outputs = train_step(model, images, labels, criterion, optimizer, device)
    
    assert loss.item() > 0
    assert torch.isfinite(loss)
    
    with torch.inference_mode():
        preds = torch.argmax(outputs, 1)
        accuracy = compute_accuracy(preds, labels.to(device))
    
    assert 0 <= accuracy <= 1


@pytest.mark.parametrize("device", ["cuda"] if torch.cuda.is_available() else ["cpu"])
def test_training(device, data_loaders):
    device = torch.device(device)
    train_loader, test_loader = data_loaders
    
    # Конфигурация с малым learning rate
    config1 = {
        "batch_size": 32,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "epochs": 1,
        "zero_init_residual": False
    }
    
    # Конфигурация с большим learning rate
    config2 = {
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "epochs": 1,
        "zero_init_residual": True
    }
    
    model1, accuracy1, metrics1 = train_model(config1, train_loader, test_loader, device)
    model2, accuracy2, metrics2 = train_model(config2, train_loader, test_loader, device)
    
    assert 0 <= accuracy1 <= 1
    assert 0 <= accuracy2 <= 1
    
    assert len(metrics1["train_losses"]) > 0
    assert len(metrics2["train_losses"]) > 0
    
    assert len(metrics1["test_accuracies"]) > 0
    assert len(metrics2["test_accuracies"]) > 0

    temp_dir = tempfile.gettempdir()
    model_path1 = os.path.join(temp_dir, "test_model1.pt")
    model_path2 = os.path.join(temp_dir, "test_model2.pt")
    
    save_model(model1, model_path1)
    save_model(model2, model_path2)
    
    assert os.path.exists(model_path1)
    assert os.path.exists(model_path2)
    
    loaded_model = create_model(device=device)
    loaded_model.load_state_dict(torch.load(model_path1))
    loaded_model.eval()
    
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    with torch.inference_mode():
        output = loaded_model(dummy_input)
    
    assert output.shape == (1, 10)

def test_train():
    os.system("py train.py")

    assert os.path.exists("model.pt"), "Файл модели не сохранен"

    os.system("py compute_metrics.py  ")

    assert os.path.exists("final_metrics.json"), "Файл с метриками не сохранен"

    with open("final_metrics.json", "r") as f:
        metrics = json.load(f)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1