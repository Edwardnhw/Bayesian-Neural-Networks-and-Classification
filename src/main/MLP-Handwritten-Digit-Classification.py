


# %%
'''
Question 2.0 Skeleton Code

Here you should load the data and plot
the means for each of the digit classes.
'''
import sys
import os

sys.path.append(os.path.abspath("src/utilities"))

import data
import numpy as np
import matplotlib.pyplot as plt

def plot_means(train_data, train_labels):
    """
    Plots the mean images for each digit class (0-9) side by side.
    
    Args:
        train_data (np.ndarray): The training images, each image being a vector of size 64.
        train_labels (np.ndarray): The labels for each training image.
    """
    means = []
    
    # Iterate over each digit class (0 to 9)
    for i in range(10):
        # Retrieve all images of the current digit class
        i_digits = data.get_digits_by_label(train_data, train_labels, i)
        
        # Calculate the mean for the digit class and reshape to 8x8
        mean_digit = np.mean(i_digits, axis=0).reshape((8, 8))
        means.append(mean_digit)
    
    # Concatenate all mean images horizontally for a side-by-side display
    all_concat = np.concatenate(means, axis=1)
    
    # Plotting
    plt.imshow(all_concat, cmap='gray')
    plt.title("Mean Images for Each Digit Class (0-9)")
    plt.axis('off')  # Hide axis for clearer visualization
    plt.show()

if __name__ == '__main__':
    # Load the data
    train_data, train_labels, _, _ = data.load_all_data_from_zip('data/a3digits.zip', 'data')
    plot_means(train_data, train_labels)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import data  # Assuming data.py provides necessary data loading functions

# Configuration
input_size = 64  # Number of input features (8x8 image, flattened)
output_size = 10  # Number of classes (digits 0-9)
Kw = 64  # Width of each hidden layer (used for part (i))
num_epochs = 10  # Number of training epochs
batch_size = 64  # Batch size
learning_rate = 0.001  # Learning rate

# Load data
train_data, train_labels, test_data, test_labels = data.load_all_data_from_zip('a3digits.zip', 'data')
train_data = torch.tensor(train_data, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# DataLoader for batching
train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=False)

# MLP Model Class
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_width, depth):
        super(MLP, self).__init__()
        layers = []
        
        # Input layer to the first hidden layer if depth > 0
        if depth > 0:
            layers.append(nn.Linear(input_size, hidden_width))
            layers.append(nn.ReLU())
        
        # Add additional hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())
        
        # Final output layer
        layers.append(nn.Linear(hidden_width if depth > 0 else input_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Training and evaluation function
def train_and_evaluate(depth, hidden_width):
    model = MLP(input_size, output_size, hidden_width, depth)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Training accuracy
        correct_train, total_train = 0, 0
        for images, labels in train_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        train_accuracy = 100 * correct_train / total_train

        # Testing accuracy
        correct_test, total_test = 0, 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
        test_accuracy = 100 * correct_test / total_test

    return train_accuracy, test_accuracy

# Part (i): Varying depth from 0 to 10 with fixed width Kw
depth_accuracies = []
for depth in range(0, 11):
    train_acc, test_acc = train_and_evaluate(depth, Kw)
    depth_accuracies.append((depth, train_acc, test_acc))
    print(f"Depth: {depth}, Train Accuracy(%): {train_acc:.4f}, Test Accuracy(%): {test_acc:.4f}")

# Plotting results for Part (i)
depths, train_accuracies, test_accuracies = zip(*depth_accuracies)
plt.figure(figsize=(8, 6))
plt.plot(depths, train_accuracies, label="Train Accuracy")
plt.plot(depths, test_accuracies, label="Test Accuracy")
plt.xlabel("Network Depth")
plt.ylabel("Accuracy (%)")
plt.title("Train and Test Accuracy vs. Network Depth")
plt.legend()
plt.show()

# Part (ii): Varying width from 16 to 256 with fixed depth Kd
Kd = 3  # Example fixed depth for Part (ii)
width_accuracies = []
for width in [16, 32, 64, 128, 256]:
    train_acc, test_acc = train_and_evaluate(Kd, width)
    width_accuracies.append((width, train_acc, test_acc))
    print(f"Width: {width}, Train Accuracy(%): {train_acc:.4f}, Test Accuracy(%): {test_acc:.4f}")

# Plotting results for Part (ii)
widths, train_accuracies, test_accuracies = zip(*width_accuracies)
plt.figure(figsize=(8, 6))
plt.plot(widths, train_accuracies, label="Train Accuracy")
plt.plot(widths, test_accuracies, label="Test Accuracy")
plt.xlabel("Network Width")
plt.ylabel("Accuracy (%)")
plt.title("Train and Test Accuracy vs. Network Width")
plt.legend()
plt.show()


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
import numpy as np
import data 

# Configuration
dropout_options = np.arange(0.0, 0.6, 0.1)  # Dropout rates to evaluate

# Define MLP with Dropout
class MLPWithDropout(nn.Module):
    def __init__(self, input_size, output_size, hidden_width, depth, dropout_rate):
        super(MLPWithDropout, self).__init__()
        layers = []
        
        # Input layer to first hidden layer
        layers.append(nn.Linear(input_size, hidden_width))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Final output layer
        layers.append(nn.Linear(hidden_width, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Cross-validation function
def cross_validate_dropout(dropout_rate, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_train_accuracies, fold_val_accuracies = [], []
    
    for train_idx, val_idx in kf.split(train_data):
        # Split train and validation data for this fold
        train_fold = Subset(TensorDataset(train_data, train_labels), train_idx)
        val_fold = Subset(TensorDataset(train_data, train_labels), val_idx)
        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = MLPWithDropout(input_size, output_size, Kw, 5, dropout_rate)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        for epoch in range(num_epochs):
            model.train()
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Evaluation on training fold and validation fold
        model.eval()
        train_accuracy = calculate_accuracy(model, train_loader)
        val_accuracy = calculate_accuracy(model, val_loader)
        fold_train_accuracies.append(train_accuracy)
        fold_val_accuracies.append(val_accuracy)
    
    # Average accuracies across folds
    avg_train_accuracy = np.mean(fold_train_accuracies)
    avg_val_accuracy = np.mean(fold_val_accuracies)
    
    return avg_train_accuracy, avg_val_accuracy

def calculate_accuracy(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Find the best dropout rate
best_dropout, best_val_accuracy = 0.0, 0.0
dropout_results = {}

for dropout_rate in dropout_options:
    avg_train_accuracy, avg_val_accuracy = cross_validate_dropout(dropout_rate)
    dropout_results[dropout_rate] = (avg_train_accuracy, avg_val_accuracy)
    
    print(f"Dropout: {dropout_rate:.1f}, Train Accuracy: {avg_train_accuracy:.2f}%, Validation Accuracy: {avg_val_accuracy:.2f}%")
    
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        best_dropout = dropout_rate

print(f"\nOptimal Dropout Rate: {best_dropout:.1f} with Validation Accuracy: {best_val_accuracy:.2f}%")

# Train and evaluate final model on test data
final_model = MLPWithDropout(input_size, output_size, Kw, 5, best_dropout)
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Full training on training data with optimal dropout
for epoch in range(num_epochs):
    final_model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = final_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Calculate final accuracies
train_accuracy = calculate_accuracy(final_model, train_loader)
test_accuracy = calculate_accuracy(final_model, test_loader)
print(f"\nFinal Model with Dropout {best_dropout:.1f} - Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")


# %%
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier

# Load data and best model from previous step
best_dropout = 0.0 
final_model = MLPClassifier(hidden_layer_sizes=(64,)*5, max_iter=500, random_state=42)
train_data_dropout = train_data * np.random.binomial(1, 1 - best_dropout, size=train_data.shape)
final_model.fit(train_data_dropout, train_labels)

# Evaluate predictions on test data
test_preds = final_model.predict(test_data)

# Metric 1: ROC Curve (for multiclass, plot one curve per class)
test_labels_binarized = label_binarize(test_labels, classes=range(10))
test_preds_proba = final_model.predict_proba(test_data)

plt.figure(figsize=(10, 8))
for i in range(10):
    fpr, tpr, _ = roc_curve(test_labels_binarized[:, i], test_preds_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title("Multiclass ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

# Metric 2: Confusion Matrix
conf_matrix = confusion_matrix(test_labels, test_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Metric 3: Accuracy
accuracy = accuracy_score(test_labels, test_preds)
print(f"Accuracy: {accuracy:.4f}")

# Metric 4: Precision
precision = precision_score(test_labels, test_preds, average='macro')
print(f"Precision: {precision:.4f}")

# Metric 5: Recall
recall = recall_score(test_labels, test_preds, average='macro')
print(f"Recall: {recall:.4f}")


