# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, device, epochs=5):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")


    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

    return model, train_losses