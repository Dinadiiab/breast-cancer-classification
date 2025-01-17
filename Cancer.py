# Import necessary libraries
import pandas as pd  # Version 1.3.5
import numpy as np  # Version 1.21.4
import matplotlib.pyplot as plt  # Version 3.5.0
import seaborn as sns  # Version 0.11.2
from sklearn.model_selection import train_test_split  # Version 0.24.2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf  # Version 2.7.0
from tensorflow.keras.callbacks import EarlyStopping
from ucimlrepo import fetch_ucirepo  # Ensure the latest version is installed

# Python version: 3.9

# Step 1: Fetch dataset
# Using the UCI Machine Learning Repository's API to fetch the Breast Cancer Wisconsin Diagnostic dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# Extract features (X) and target (y) from the dataset
X = breast_cancer_wisconsin_diagnostic.data.features  # Features are numerical values of tumor measurements
y = breast_cancer_wisconsin_diagnostic.data.targets  # Targets represent tumor diagnosis (Malignant or Benign)

# Display dataset metadata (optional, for reference only)
print("Dataset Metadata:")
print(breast_cancer_wisconsin_diagnostic.metadata)

# Step 2: Preprocess the data
# Normalize features using StandardScaler to mean=0 and variance=1
# This helps in improving the performance of the model by standardizing feature values
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)

# Encode labels: 0 for benign, 1 for malignant
# Label encoding converts the target labels to a numerical format suitable for the model
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Step 3: Split data into training and validation sets
# Splitting ensures we have separate data for training the model and validating its performance
# 80% of the data is used for training, and 20% for validation/testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Define the model
# Building a simple neural network with input, hidden, and output layers
# The architecture includes one input layer, one hidden layer, and one output layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=16, activation='relu', input_shape=(X.shape[1],)),  # Input layer with 16 neurons
    tf.keras.layers.Dense(units=32, activation='relu'),  # Hidden layer with 32 neurons
    tf.keras.layers.Dropout(0.3),  # Dropout layer to prevent overfitting during training
    tf.keras.layers.Dense(units=1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
])

# Step 5: Compile the model
# Compiling configures the model for training by defining the optimizer, loss function, and evaluation metrics
# Adam optimizer is used for adaptive learning rate, and binary crossentropy is suited for binary classification tasks
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary for reference
# This provides an overview of the model architecture and total trainable parameters
model.summary()

# Step 6: Train the model
# Early stopping callback to prevent overfitting
# Training stops early if validation loss does not improve for 2 consecutive epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model with validation data
# Validation data is used during training to monitor the model's performance on unseen data
history = model.fit(
    x_train, y_train,
    epochs=11,  # Training for up to 50 epochs (stops earlier if early stopping is triggered)
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
    verbose=1  # Verbose output to track training progress
)

# Step 7: Evaluate the model
# Test performance on validation data
# Evaluates the model on the test set and prints the loss and accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Step 8: Make predictions
# Predict probabilities for test data using the trained model
predictions = model.predict(x_test)

# Convert probabilities to binary predictions (threshold = 0.5)
# Predictions above 0.5 are classified as malignant, and below 0.5 as benign
predicted_classes = (predictions > 0.5).astype(int)

# Step 9: Visualize Results
# Confusion Matrix
# The confusion matrix provides insights into the number of true positives, false positives, true negatives, and false negatives
cm = confusion_matrix(y_test, predicted_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
# The classification report shows precision, recall, F1-score, and support for each class
print("\nClassification Report:")
print(classification_report(y_test, predicted_classes, target_names=label_encoder.classes_))

# Training and Validation Loss
# Plotting the training and validation loss over epochs helps visualize convergence and overfitting
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Training and Validation Accuracy
# Plotting the training and validation accuracy over epochs helps visualize model performance
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 10: Save the model
# Save the trained model for reuse
# The model is saved in HDF5 format, which can be loaded later for predictions or further training
model.save('breast_cancer_classification_model.h5')
print("Model saved as 'breast_cancer_classification_model.h5'")

# Save the scaler and label encoder for preprocessing new data
# The scaler and label encoder are saved using joblib for consistency during deployment
import joblib
joblib.dump(standard_scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
print("Scaler and Label Encoder saved.")
