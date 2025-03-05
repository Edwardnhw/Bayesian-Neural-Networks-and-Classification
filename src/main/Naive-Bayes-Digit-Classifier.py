
# %%
'''
Naive Bayes Classifier Implementation
'''
import sys
import os

sys.path.append(os.path.abspath("src/utilities"))

import data
import numpy as np
import matplotlib.pyplot as plt

def binarize_data(pixel_values):
    '''
    (a) Convert real-valued features into binary features using a threshold of 0.5.
    '''
    return np.where(pixel_values > 0.5, 1.0, 0.0)

def compute_parameters(train_data, train_labels):
    '''
    (b) Compute the eta MAP estimate using augmented data as prior.
    Returns a numpy array of shape (10, 64) where the ith row corresponds to the
    eta for the ith digit class.
    '''
    alpha = beta = 2  # Beta prior parameters
    eta = np.zeros((10, 64))
    
    for k in range(10):
        class_data = train_data[train_labels == k]
        # Apply the Beta prior by adding pseudocounts of 1 for each pixel being ON and OFF
        eta[k] = (np.sum(class_data, axis=0) + alpha - 1) / (class_data.shape[0] + alpha + beta - 2)
        
    return eta

def plot_images(class_images):
    '''
    (c) Plot each eta vector as an 8x8 grayscale image side by side for each class.
    '''
    fig, axes = plt.subplots(1, 10, figsize=(15, 5))
    for i in range(10):
        img = class_images[i].reshape((8, 8))
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Class {i}')
        axes[i].axis('off')
    plt.show()

def generate_new_data(eta):
    '''
    (d) Generate a new data point for each class by sampling from Bernoulli distribution with parameter eta.
    '''
    generated_data = np.zeros((10, 64))
    for k in range(10):
        generated_data[k] = np.random.binomial(1, eta[k])
    plot_images(generated_data)

def generative_likelihood(bin_digits, eta):
    '''
    (e) Compute the generative log-likelihood log p(x|y, eta).
    Returns an (n, 10) numpy array where each entry [i, k] is log p(x^(i)|y=k, eta).
    '''
    n = bin_digits.shape[0]
    gen_likelihood = np.zeros((n, 10))
    
    for k in range(10):
        # log p(x|y=k) = sum_j [x_j log(eta_kj) + (1 - x_j) log(1 - eta_kj)]
        gen_likelihood[:, k] = np.sum(bin_digits * np.log(eta[k]) + (1 - bin_digits) * np.log(1 - eta[k]), axis=1)
        
    return gen_likelihood

def conditional_likelihood(bin_digits, eta):
    '''
    (f) Compute the conditional log-likelihood log p(y|x, eta).
    Returns an (n, 10) numpy array with the posterior log-probabilities for each class.
    '''
    gen_likelihood = generative_likelihood(bin_digits, eta)
    log_prior = np.log(1 / 10)  # p(y=k) = 1/10 for all classes
    log_posterior = gen_likelihood + log_prior
    log_posterior -= np.logaddexp.reduce(log_posterior, axis=1, keepdims=True)  # Normalize using log-sum-exp trick
    
    return log_posterior

def avg_conditional_likelihood(bin_digits, labels, eta):
    '''
    (e) Compute the average conditional likelihood over the true class labels.
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    labels = labels.astype(int)  # Ensure labels are integers
    correct_class_likelihoods = cond_likelihood[np.arange(len(labels)), labels]
    avg_likelihood = np.mean(correct_class_likelihoods)
    return avg_likelihood


def classify_data(bin_digits, eta):
    '''
    (f) Classify data points by choosing the most likely posterior class.
    '''
    cond_likelihood = conditional_likelihood(bin_digits, eta)
    return np.argmax(cond_likelihood, axis=1)

def main():
    # Load and binarize data
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    train_data, test_data = binarize_data(train_data), binarize_data(test_data)

    # (b) Fit the model
    eta = compute_parameters(train_data, train_labels)

    # (c) Plot each eta vector as an 8x8 image
    plot_images(eta)

    # (d) Generate new data points for each class and plot
    generate_new_data(eta)

    # (e) Compute and print the average conditional log-likelihood for train and test sets
    train_avg_likelihood = avg_conditional_likelihood(train_data, train_labels, eta)
    test_avg_likelihood = avg_conditional_likelihood(test_data, test_labels, eta)
    print(f"Average conditional log-likelihood (train): {train_avg_likelihood}")
    print(f"Average conditional log-likelihood (test): {test_avg_likelihood}")

    # (f) Classify train and test data and calculate accuracy
    train_predictions = classify_data(train_data, eta)
    test_predictions = classify_data(test_data, eta)
    train_accuracy = np.mean(train_predictions == train_labels)
    test_accuracy = np.mean(test_predictions == test_labels)
    print(f"Train accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()



