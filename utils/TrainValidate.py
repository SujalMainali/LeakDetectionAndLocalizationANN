import torch

# Training function
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
        optimizer.zero_grad()  # Zero gradients from the previous step

        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, targets)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()

    return total_loss / len(dataloader)


# Validation function
def validate_model(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Calculate loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)