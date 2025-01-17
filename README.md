# Breast Cancer Classification Using Neural Networks

This project is a machine learning pipeline that classifies breast cancer tumors as benign or malignant using the "Breast Cancer Wisconsin (Diagnostic)" dataset. The project utilizes Python and TensorFlow to train a neural network for binary classification.

## Dataset Information
- **Source**: UCI Machine Learning Repository
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic)
- **Link**: [Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- **Features**: 30 real-valued features describing tumor characteristics
- **Target**: Diagnosis (M = malignant, B = benign)
- **Instances**: 569 samples

## Project Overview
The goal is to build a binary classification model that can accurately classify tumors as malignant or benign based on the provided features.

## Tools and Libraries
The following libraries are used in this project:
- **Python**: 3.9
- **Pandas**: 1.3.5
- **NumPy**: 1.21.4
- **Matplotlib**: 3.5.0
- **Seaborn**: 0.11.2
- **Scikit-learn**: 0.24.2
- **TensorFlow**: 2.7.0
- **Joblib**: 1.1.0

## Steps in the Project
### 1. Data Fetching
The dataset is fetched using the `ucimlrepo` library, which provides direct access to UCI datasets.

### 2. Data Preprocessing
- **Standardization**: The features are normalized to have zero mean and unit variance using `StandardScaler`.
- **Label Encoding**: The target labels are encoded into binary values (0 for benign, 1 for malignant).

### 3. Data Splitting
The dataset is split into training (80%) and testing (20%) subsets using `train_test_split`.

### 4. Neural Network Model
The model is a feedforward neural network with the following architecture:
- Input Layer: 16 neurons
- Hidden Layer: 32 neurons with ReLU activation and L2 regularization
- Dropout Layer: Dropout rate of 0.3 to prevent overfitting
- Output Layer: 1 neuron with sigmoid activation for binary classification

### 5. Training and Validation
The model is trained using the Adam optimizer and binary crossentropy loss for up to 50 epochs with early stopping to prevent overfitting.

### 6. Evaluation
- The model is evaluated using accuracy and loss metrics.
- A confusion matrix and classification report provide further insights into the model’s performance.

### 7. Visualization
Plots of training and validation loss and accuracy over epochs are generated to monitor the training process.

### 8. Model Saving
- The trained model is saved in HDF5 format (`breast_cancer_classification_model.h5`).
- The `StandardScaler` and `LabelEncoder` are saved using `joblib` for consistent preprocessing of new data.

## Results
- **Test Accuracy**: 97.37%
- **Confusion Matrix**:
  | Actual \ Predicted | Benign (B) | Malignant (M) |
  |-------------------|------------|---------------|
  | **Benign (B)**    | 67         | 0             |
  | **Malignant (M)** | 3          | 44            |

- **Precision, Recall, F1-Score**:
  | Class       | Precision | Recall | F1-Score |
  |-------------|-----------|--------|----------|
  | **Benign**  | 0.96      | 1.00   | 0.98     |
  | **Malignant** | 1.00    | 0.94   | 0.97     |

## Repository Structure
```
project/
├── data/                # If including sample datasets
├── models/              # Saved models
├── scripts/             # Python code files
├── results/             # Output plots and results
├── README.md            # Project overview
├── requirements.txt     # Dependency list
```

## How to Run the Code
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/breast-cancer-classification.git
   cd breast-cancer-classification
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Script**:
   ```bash
   python script_name.py
   ```



## Acknowledgments
- The dataset creators: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street.
- UCI Machine Learning Repository for hosting the dataset.

