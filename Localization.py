import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.Dataset import LeakDataset
from utils.TrainValidate import train_model, validate_model
from utils.SaveLoad import save_model_with_params


class LeakLocalizationNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=3):
        """
        Initialize the neural network.

        Parameters:
        - input_dim (int): Number of input features (size of the input vector).
        - hidden_dims (list of int): List containing the number of neurons in each hidden layer.
        - output_dim (int): Number of output variables. Default is 3 (X_coor, Y_coor, burst_size).
        """
        super(LeakLocalizationNN, self).__init__()
        self.input_dim = input_dim

        # Define the layers
        self.hidden_layers = nn.ModuleList()
        
        # Input to first hidden layer
        self.hidden_layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Subsequent hidden layers
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Last layer (hidden to output)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Forward pass through all hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer (predict X_coor, Y_coor, burst_size)
        x = self.output_layer(x)
        return x

if __name__ == "__main__":
    input_dim = 10  # Number of pressure head readings per input sample
    hidden_dims = [45, 40, 45]  # Custom hidden layers configuration
    output_dim = 3 

    csv_file = "leak_data.csv"  # Path to your CSV file
    input_columns = ["Node1", "Node2", "Node3", "Node4", "Node5"]  # Define input columns
    output_columns = ["X_coor", "Y_coor", "burst_size"]  # Define output columns

    # Create dataset
    dataset = LeakDataset(csv_file, input_columns, output_columns)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeakLocalizationNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 20
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    # Save the trained model with normalization parameters
    normalization_params = dataset.get_normalization_params()

    save_model_with_params(
        model=model,
        filepath="models/leak_localization_with_norm_params.pth",
        input_means=normalization_params["input_means"],
        input_stds=normalization_params["input_stds"],
        output_means=normalization_params["output_means"],
        output_stds=normalization_params["output_stds"],
    )

