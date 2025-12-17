import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load dataset
def load_dataset(csv_file, input_columns, output_column):
    """
    Load the dataset from a CSV file.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - input_columns (list of str): List of column names for inputs.
    - output_column (str): Column name for the target output.

    Returns:
    - X (pd.DataFrame): Input features.
    - y (pd.Series): Target output.
    """
    data = pd.read_csv(csv_file)
    X = data[input_columns]
    y = data[output_column]
    return X, y


# Train Random Forest
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42):
    """
    Train a Random Forest Classifier.

    Parameters:
    - X_train (pd.DataFrame): Training input features.
    - y_train (pd.Series): Training target output.
    - n_estimators (int): Number of trees in the forest. Default: 100.
    - max_depth (int): Maximum depth of the trees. Default: None (expand until pure).
    - random_state (int): Random seed for reproducibility. Default: 42.

    Returns:
    - model (RandomForestClassifier): Trained random forest model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    return model


# Evaluate Random Forest
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.

    Parameters:
    - model (RandomForestClassifier): Trained random forest model.
    - X_test (pd.DataFrame): Test input features.
    - y_test (pd.Series): Test target output.

    Prints:
    - Confusion matrix.
    - Classification report.
    - Accuracy.
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy Score:")
    print(accuracy_score(y_test, y_pred))


# Pipeline
if __name__ == "__main__":
    # CSV file path
    csv_file = "leak_data.csv"

    # Feature and target columns
    input_columns = ["Node1", "Node2", "Node3", "Node4", "Node5"]  # Input features (pressure at nodes)
    output_column = "Leak"  # Target (0 = no leak, 1 = leak)

    # Load dataset
    X, y = load_dataset(csv_file, input_columns, output_column)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save trained model
    joblib.dump(model, "models/random_forest_leak_detection.pkl")
    print("\nRandom Forest model saved at 'models/random_forest_leak_detection.pkl'")