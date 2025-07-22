import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

device = torch.device("cuda")
lr = [0.1, 0.001, 0.0001]
size = 32

trainer = transforms.Compose([
    transforms.RandomResizedCrop((224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

second_trainer = transforms.Compose([
    transforms.Resize((256)),
    transforms.CenterCrop((224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
path = "../Dataset"
all_datasets = datasets.ImageFolder(root=path, transform=None)

totals = len(all_datasets)
train_size = int(0.64*totals)
val_size = int(0.16*totals)
test_size = totals - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(all_datasets, [train_size, val_size, test_size])

train_dataset.dataset.transform = trainer
val_dataset.dataset.transform = second_trainer
test_dataset.dataset.transform = second_trainer

load1 = DataLoader(train_dataset, batch_size=size, shuffle=True, num_workers=4)
load2 = DataLoader(val_dataset, batch_size=size)
load3 = DataLoader(test_dataset, batch_size=size)

for l in lr:
    print(f"learning rate is {l}")
    epochs = 10
    model = models.resnet50(pretrained=True)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=l)
    patience = 5
    loss = float('inf')
    trigger = 0

    for epoch in range(epochs):
        model.train()
        lo = 0.0
        for i, l in load1:
            i, l = i.to(device), l.to(device)
            optimizer.zero_grad()
            outputs = model(i)
            loss = criterion(outputs, l)
            loss.backward()
            optimizer.step()
            lo += loss.item()
        lo /= len(load1)
        model.eval()
        valLoss = 0.0
        right = 0
        total = 0
        with torch.no_grad():
            for i, l in load2:
                i, l = i.to(device), l.to(device)
                outputs = model(i)
                loss = criterion(outputs, l)
                valLoss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += l.size(0)
                right += (predicted == l).sum().item()
        valLoss /= len(load2)
        acc = right / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {lo:.4f}, Val Loss: {valLoss:.4f}, Accuracy: {acc:.4f}")
        if valLoss < loss:
            loss = valLoss
            trigger = 0
            torch.save(model.state_dict(), f"model_{l}.pth")
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping")
                break
    model.load_state_dict(torch.load(f"model_{l}.pth"))
    model.eval()
    test_loss = 0.0
    right = 0
    total = 0
    with torch.no_grad():
        for i, l in load3:
            i, l = i.to(device), l.to(device)
            outputs = model(i)
            loss = criterion(outputs, l)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += l.size(0)
            right += (predicted == l).sum().item()
    test_loss /= len(load3)
    acc = right / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.4f}")
