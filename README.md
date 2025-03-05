# Naive Bayes and Gaussian Classifier for Digit Classification

**Author**: Hon Wa Ng\
**Date**: October 2024  

## Overview

This project explores Bayesian methods and neural networks for handwritten digit classification. It implements multiple classification approaches, including:

- Naïve Bayes Classifier
- Conditional Gaussian Classifier
- Multilayer Perceptron (MLP)
The project evaluates model performance on an 8x8 pixel grayscale handwritten digit dataset, utilizing generative models and deep learning-based approaches.

## Objectives

- Implement and compare Naïve Bayes, Conditional Gaussian, and MLP classifiers.
- Evaluate classification accuracy for different models.
- Explore Bayesian approaches in digit classification.
- Apply cross-validation and hyperparameter tuning for optimization.


## Repository Structure
```bash
BAYESIAN-NEURAL-NETWORKS-AND-DIGIT-CLASSIFICATION/
│── data/                              # Dataset storage
│   ├── a3digits/                       # Folder containing digit dataset
│   ├── a3digits.zip                  # Compressed dataset
│   ├── data.py                          # Data loading utilities
│
│── docs/                               # Documentation files
│   ├── bayesian_digit_classification_analysis.pdf   # Project analysis
│   ├── problem_statement.pdf            # Task description and dataset details
│
│── src/                                # Source code
│   ├── main/                            # Main scripts for different models
│   │   ├── Conditional-Gaussian-Digit-Classifier.py   # Conditional Gaussian classifier
│   │   ├── Naive-Bayes-Digit-Classifier.py            # Naive Bayes classifier
│   │   ├── MLP-Handwritten-Digit-Classification.py    # MLP-based digit classification
│   ├── utilities/                        # Additional utility functions
│   │   ├── data.py                         # Data processing helper functions
│
│── LICENSE                             # MIT License
│── README.md                            # Project documentation
```

---

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/Edwardnhw/NaiveBayes-Gaussian-Classifier.git
cd NaiveBayes-Gaussian-Classifier

```

### 2. Install Dependencies
Ensure you have Python installed (>=3.7), then run:
```
pip install -r requirements.txt

```

### 3. Run Classifiers
## Run the Conditional Gaussian Digit Classifier

```
python src/main/Conditional-Gaussian-Digit-Classifier.py

```


## Run the Naive Bayes Classifier

```
python src/main/Naive-Bayes-Digit-Classifier.py

```

## Run the MLP-based Digit Classifier

```
python src/main/MLP-Handwritten-Digit-Classification.py

```

---
## Methods Used

1. Naïve Bayes Classifier
- Binarizes input images.
- Uses a Bayesian probabilistic model with Laplace smoothing.
- Computes posterior probabilities to classify digits.
2. Conditional Gaussian Classifier
- Models feature distributions as multivariate Gaussians per digit class.
- Computes mean vectors and covariance matrices for each class.
- Uses log-likelihood estimation for classification.
3. Multilayer Perceptron (MLP)
- Implements a fully connected neural network.
- Uses ReLU activation and softmax classification.
- Trained using Adam optimizer and Cross-Entropy Loss.
- Evaluates model performance with cross-validation.

---

## Results & Analysis

- Classification performance is evaluated using accuracy, log-likelihood scores, and confusion matrices.
- Performance is compared across Naïve Bayes, Gaussian Classifier, and MLP models.
- Hyperparameter tuning explores hidden layer size, dropout rates, and training depth.

Refer to the bayesian_digit_classification_analysis.pdf in the docs/ folder for detailed results.


---
## License
This project is licensed under the MIT License.



