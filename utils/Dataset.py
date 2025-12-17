import pandas as pd
import torch
from torch.utils.data import Dataset

class LeakDataset(Dataset):
    def __init__(self, csv_file, input_columns, output_columns):
        """
        Initializes the dataset from a CSV file.
        
        Parameters:
        - csv_file (str): Path to the CSV file.
        - input_columns (list of str): List of column names to use as inputs.
        - output_columns (list of str): List of column names to use as outputs.
        """
        # Load dataset
        data = pd.read_csv(csv_file)
        
        # Extract inputs (features) and outputs (targets)
        self.inputs = data[input_columns].values  # Convert to numpy array
        self.outputs = data[output_columns].values  # Convert to numpy array
        
        # Convert to PyTorch tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.outputs = torch.tensor(self.outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]