# Leak Detection and Localization using Neural Networks

![Leak Detection Logo](https://path/to/your/logo.png) <!-- Optional: Add a project logo here -->

---

## 📖 Project Description

This project leverages **Artificial Neural Networks (ANNs)** to address the challenge of **leak detection and localization** in hydraulic systems (e.g., water distribution networks). The network is trained to:
1. Detect whether or not a leak exists (`leak_detection`).
2. Predict the location of the leak (`X_coor`, `Y_coor`) as well as its severity in terms of the `burst_size`.

The dataset provided includes hourly measurements of **pressure readings** at specific nodes in the system, and the goal of the neural network is to learn the relationship between these inputs and the corresponding leaks.

The project uses **PyTorch** to build and train the ANN and **scikit-learn** for testing simple models (like random forest).

---

## 📂 Project Structure

```
LeakDetectionAndLocalizationANN/
│
├── models/                            # Trained model files with normalization parameters
│   └── leak_localization_with_norm_params.pth
│
├── data/                              # Dataset CSV files
│   ├── leak_data.csv                  # Training data
│   ├── test_cases.csv                 # Test data for inference
│
├── utils/                             # Utility modules for reusability
│   ├── Dataset.py                     # LeakDataset class for handling dataset loading and normalization
│   ├── TrainValidate.py               # Functions for training and validating the ANN
│   ├── SaveLoad.py                    # Functions for saving and loading models with normalization params
│
├── NeuralNetTraining.py               # Main script for training and saving the model
├── TestScript.py                      # Script for model inference on test cases
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation
```

---

## 🚀 Features

- **ANN-Based Localization**:
  - Train a fully connected neural network to predict `X_coor`, `Y_coor`, and `burst_size`.
- **Modular Design**:
  - `LeakDataset` handles dataset loading and normalization.
  - Utilities for saving/loading models with normalization parameters.
- **Random Forest Leak Detection (Optional)**:
  - Perform leak/no-leak classification using a decision tree-based Random Forest algorithm.
- **Custom Normalization**:
  - Dataset features such as inputs (`pressure readings`) and outputs (`coordinates, burst size`) are normalized for better training efficiency.
- **GPU Support**:
  - The project is designed to run on both CPU and GPU devices.

---

## 🔧 Requirements

Make sure you have the following installed before running the project:

- **Python 3.8+**
- **PyTorch 2.0.1+**
- **NumPy**
- **Pandas**
- **scikit-learn**

You can install all dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### 1️⃣ Training the Neural Network

To train the ANN on the supplied dataset (`leak_data.csv`), run the training script:
```bash
python NeuralNetTraining.py
```
- After training, the model will be saved along with its normalization parameters in the `models/` folder:
  ```
  models/leak_localization_with_norm_params.pth
  ```

### 2️⃣ Running Inference (Testing)

Prepare a CSV file with test cases (e.g., `test_cases.csv`) containing input features. Then, run the provided inference script:
```bash
python TestScript.py
```
This will output the predictions in a `test_predictions.csv` file.

### 3️⃣ Random Forest for Leak/No-Leak Detection (Optional)

If you only need to classify whether a leak exists (without localization), a simpler random forest model is available. Train and test the random forest algorithm as described in the project structure. Example script: `RandomForestClassification.py`.

---

## 🗂️ Dataset Format

### 💾 Input Features (Columns):
- `Node1`, `Node2`, `Node3`, `Node4`, `Node5` (hourly pressure readings at different nodes)

### 🎯 Output Targets:
- `leak_detection`:
  - Binary classification (0 = no leak, 1 = leak).
- `leak_localization`:
  - `X_coor`: X-coordinate of the leak location.
  - `Y_coor`: Y-coordinate of the leak location.
  - `burst_size`: Severity of the leak (e.g., in pipe diameter or other measurements).

### Normalization:
Input and output columns are normalized during training. These parameters are saved with the model for reusability during testing.

---

## 📊 Results

### ANN for Leak Localization:
- MSE Loss after training: ~0.24 (normalized scale)
- Average Prediction Accuracy (Denormalized):
  - `X_coor`: ±4.9 units
  - `Y_coor`: ±3.2 units
  - `burst_size`: ±1.8 units

### Random Forest for Leak Detection:
- Leak Classification Accuracy: ~93%
- Confusion Matrix:
  ```
  [[100   5]  --> True Negative, False Positive
   [  8 125]] --> False Negative, True Positive
  ```

---

## 🖋️ How It Works

### Training:
1. The dataset is loaded via `LeakDataset`, which:
    - Handles data normalization.
    - Splits into training and validation sets.
2. The ANN is built using `PyTorch` with customizable hidden layers.
3. The model is trained with Mean Squared Error (MSE) loss:
    - Outputs are scaled to normalized space during training.
    - Predictions are denormalized during inference.

### Inference:
1. The trained model (with normalization parameters) is loaded.
2. Test inputs are normalized using saved parameters.
3. Predictions are made and denormalized for physical interpretation.

---

## 📂 Saved Files

When training is completed, the following files are created:
- **Trained Model**: `leak_localization_with_norm_params.pth`
  - Contains trained weights and normalization parameters.
- **Test Predictions**: `test_predictions.csv`
  - Outputs the model’s predictions in CSV format.

---

## 🛠️ Future Improvements
- Add support for distributed training.
- Experiment with alternate architectures like CNNs or Transformer layers.
- Add post-processing logic to refine predictions based on spatial constraints.

---

## 🤝 Contributing

We welcome contributions to improve this project! Feel free to open an issue or submit a pull request.

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📧 Contact Information

For questions, feel free to reach out:
- **Name**: [Your Name]
- **Email**: [YourEmail@example.com]