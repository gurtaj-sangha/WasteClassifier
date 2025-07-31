import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = [0.1, 0.001, 0.0001]
size = 32

trainer = transforms.Compose([ # Training Pipeline
    transforms.RandomResizedCrop((224)), # Data Augmentation
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Convert to pytorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

second_trainer = transforms.Compose([ # Validation/test pipeline
    transforms.Resize((256)),
    transforms.CenterCrop((224)), #ResNet is trained on 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def main():
    path = "Dataset"
    all_datasets = datasets.ImageFolder(root=path, transform=None) # Load dataset

    # Split into train/val/test
    totals = len(all_datasets)
    train_size = int(0.64 * totals)
    val_size = int(0.16 * totals)
    test_size = totals - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(all_datasets, [train_size, val_size, test_size])
    # Applying transformation from pipeline
    train_dataset.dataset.transform = trainer
    val_dataset.dataset.transform = second_trainer
    test_dataset.dataset.transform = second_trainer

    load1 = DataLoader(train_dataset, batch_size=size, shuffle=True, num_workers=0)  # Set to 0 for Windows
    load2 = DataLoader(val_dataset, batch_size=size)
    load3 = DataLoader(test_dataset, batch_size=size)

    for i, learning_rate in enumerate(lr):
        print(f"\nTraining model with learning rate: {learning_rate}")
        epochs = 20

        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for p in model.parameters():
            p.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 3)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        patience = 5
        best_val_loss = float('inf')
        trigger = 0

        # Lists that store losses and accuracy
        train_losses = []
        val_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in load1:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(load1)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in load2:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            val_loss /= len(load2)
            accuracy = correct / total
            # Save stats
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(accuracy)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

            if epoch == 0 or val_loss < best_val_loss:
               
                best_val_loss = val_loss
                trigger = 0
                torch.save(model.state_dict(), f"model_{i}.pth")
            else:
                trigger += 1
                if trigger >= patience:
                    print("Early stopping")
                    break

         # Plot graphs
        epochs_range = range(1, len(train_losses) + 1)
        plt.figure(figsize=(12, 5))

        # Accuracy Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, val_accuracies, label="Validation Accuracy", color="red")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_losses, label="Train Loss", color="blue")
        plt.plot(epochs_range, val_losses, label="Validation Loss", color="orange")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.suptitle(f"Learning Rate = {learning_rate}")
        plt.tight_layout()
        plt.savefig(f"metrics_lr_{learning_rate}.png")
        plt.show()
       
    # Choose which trained model index to test (e.g., 1 = 0.001)
    test_index = 1
    print(f"\nLoading model for testing (learning rate = {lr[test_index]})")
    model.load_state_dict(torch.load(f"model_{test_index}.pth"))
    model.eval()

    test_loss = 0.0
    right = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in load3:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            right += (predicted == labels).sum().item()
    test_loss /= len(load3)
    acc = right / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {acc:.4f}")

# Required for Windows to safely run multiprocessing
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
import os
print("Saved model files:", [f for f in os.listdir() if f.endswith(".pth")])