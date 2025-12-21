import torch
import pandas as pd
from utils.SaveLoad import load_model_with_params
from ..Localization import LeakLocalizationNN

def normalize_inputs(input_data, input_means, input_stds):
    """
    Normalize the input data using means and standard deviations.

    Parameters:
    - input_data (pd.DataFrame): DataFrame containing the input features.
    - input_means (dict): Mean values for input features.
    - input_stds (dict): Std deviation values for input features.

    Returns:
    - normalized_inputs (torch.Tensor): Normalized inputs as a PyTorch Tensor.
    """
    normalized_data = input_data.copy()
    for column in input_data.columns:
        if column in input_means and column in input_stds:
            normalized_data[column] = (input_data[column] - input_means[column]) / input_stds[column]
    return torch.tensor(normalized_data.values, dtype=torch.float32)


def denormalize_outputs(predictions, output_means, output_stds):
    """
    Denormalize model predictions using means and standard deviations.

    Parameters:
    - predictions (torch.Tensor): Normalized predictions from the model.
    - output_means (dict): Mean values for output features.
    - output_stds (dict): Std deviation values for output features.

    Returns:
    - denormalized_data (pd.DataFrame): Denormalized predictions as a DataFrame.
    """
    denormalized_data = []
    for i, (key, mean) in enumerate(output_means.items()):
        std = output_stds[key]
        denormalized_predictions = predictions[:, i] * std + mean
        denormalized_data.append(denormalized_predictions.tolist())
    return pd.DataFrame({key: values for key, values in zip(output_means.keys(), denormalized_data)})


def run_test_cases(model_path, test_csv, input_columns, device):
    """
    Run inference on test cases using the trained model.

    Parameters:
    - model_path (str): Path to the saved model and normalization parameters.
    - test_csv (str): Path to the test cases CSV file.
    - input_columns (list of str): List of input feature column names.

    Returns:
    - denormalized_predictions (pd.DataFrame): Denormalized model predictions.
    """
    # Load the test data
    test_data = pd.read_csv(test_csv)
    test_inputs = test_data[input_columns]

    # Load the trained model and normalization parameters
    model = LeakLocalizationNN(input_dim=len(input_columns), hidden_dims=[45, 40, 45], output_dim=3)
    model, normalization_params = load_model_with_params(model, model_path, device)

    # Extract normalization parameters from the model
    input_means = normalization_params["input_means"]
    input_stds = normalization_params["input_stds"]
    output_means = normalization_params["output_means"]
    output_stds = normalization_params["output_stds"]

    # Normalize the inputs using saved means and stds
    normalized_inputs = normalize_inputs(test_inputs, input_means, input_stds).to(device)

    # Perform inference
    model.eval()
    with torch.no_grad():
        normalized_predictions = model(normalized_inputs)

    # Denormalize the predictions
    denormalized_predictions = denormalize_outputs(normalized_predictions, output_means, output_stds)
    return denormalized_predictions


if __name__ == "__main__":
    # Path to the saved model with normalization parameters
    model_path = "models/leak_localization_with_norm_params.pth"

    # Path to test cases CSV file
    test_csv = "test_cases.csv"  # Update this with the path to your test cases

    # Input configuration
    input_columns = ["Node1", "Node2", "Node3", "Node4", "Node5"]  # Columns in the test CSV for the inputs

    # Select device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run inference on test cases
    predictions = run_test_cases(model_path, test_csv, input_columns, device)

    # Save the predictions to a CSV file
    predictions.to_csv("test_predictions.csv", index=False)
    print("Predictions saved to 'test_predictions.csv'")