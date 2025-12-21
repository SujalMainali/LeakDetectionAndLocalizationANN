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
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    output_losses = {"X_coor": 0.0, "Y_coor": 0.0, "burst_size": 0.0}
    num_samples = 0  # Total number of samples

    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get model predictions
            outputs = model(inputs)

            # Calculate total loss
            batch_loss = criterion(outputs, targets)
            total_loss += batch_loss.item() * inputs.size(0)  # Accumulate batch loss

            # Calculate per-output loss
            for i, output_name in enumerate(output_losses.keys()):  # "X_coor", "Y_coor", ...
                output_loss = criterion(outputs[:, i], targets[:, i])
                output_losses[output_name] += output_loss.item() * inputs.size(0)

            # Update total sample count
            num_samples += inputs.size(0)

    # Normalize losses by the number of samples
    total_loss /= num_samples
    for key in output_losses:
        output_losses[key] /= num_samples

    return total_loss, output_losses