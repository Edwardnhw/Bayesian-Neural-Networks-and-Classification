
# %%
'''
Conditional Gaussian Classifier Implementation
'''

import data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class.
    Returns a numpy array of size (10, 64).
    '''
    means = np.zeros((10, 64))
    for k in range(10):  # Iterate over each class
        class_data = train_data[train_labels == k]  # Get all samples for class k
        means[k] = np.mean(class_data, axis=0)  # Compute mean across all samples for class k
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class.
    Returns a numpy array of shape (10, 64, 64).
    '''
    covariances = np.zeros((10, 64, 64))
    for k in range(10):
        class_data = train_data[train_labels == k]
        mean_k = np.mean(class_data, axis=0)
        cov_matrix = np.cov(class_data, rowvar=False)  # Compute covariance
        cov_matrix += 0.01 * np.eye(64)  # Add 0.01 * Identity for numerical stability
        covariances[k] = cov_matrix
    return covariances

def plot_cov_diagonal(covariances):
    # Plot the log-diagonal of each covariance matrix side by side
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for i in range(10):
        cov_diag_log = np.log(np.diag(covariances[i]))  # Log of diagonal elements
        cov_diag_log = cov_diag_log.reshape((8, 8))  # Reshape to 8x8
        axes[i].imshow(cov_diag_log, cmap='gray')
        axes[i].set_title(f'Class {i}')
        axes[i].axis('off')  # Hide axis
    plt.show()

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood for each class.
    Returns an n x 10 numpy array.
    '''
    n, d = digits.shape
    likelihoods = np.zeros((n, 10))
    
    for k in range(10):
        mean_k = means[k]
        cov_k = covariances[k]
        cov_inv_k = np.linalg.inv(cov_k)
        cov_det_k = np.linalg.det(cov_k)
        
        for i in range(n):
            diff = digits[i] - mean_k
            exponent = -0.5 * diff.T @ cov_inv_k @ diff
            likelihoods[i, k] = -0.5 * np.log(cov_det_k) - (d / 2) * np.log(2 * np.pi) + exponent
            
    return likelihoods

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional log-likelihood for each class.
    Returns an n x 10 numpy array.
    '''
    gen_likelihood = generative_likelihood(digits, means, covariances)
    prior = np.log(1 / 10)  # Since p(y=k) = 1/10 for all classes
    
    # Compute posterior log likelihood for each class
    log_posterior = gen_likelihood + prior
    log_posterior -= np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)  # Log-sum-exp trick for normalization
    
    return log_posterior

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels.
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    labels = labels.astype(int)  # Ensure labels are integers
    correct_class_likelihoods = cond_likelihood[np.arange(len(labels)), labels]
    avg_likelihood = np.mean(correct_class_likelihoods)
    return avg_likelihood

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class.
    Returns an array of predicted labels.
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return np.argmax(cond_likelihood, axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Plot covariance diagonal
    plot_cov_diagonal(covariances)

    # Calculate average conditional log-likelihood for train and test sets
    train_avg_likelihood = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg_likelihood = avg_conditional_likelihood(test_data, test_labels, means, covariances)
    print(f"Average conditional log-likelihood (train): {train_avg_likelihood}")
    print(f"Average conditional log-likelihood (test): {test_avg_likelihood}")

    # Classify test data and compute accuracy
    train_predictions = classify_data(train_data, means, covariances)
    test_predictions = classify_data(test_data, means, covariances)
    train_accuracy = np.mean(train_predictions == train_labels)
    test_accuracy = np.mean(test_predictions == test_labels)
    print(f"Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()



