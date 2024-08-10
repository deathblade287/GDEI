import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Function to generate synthetic data


def generate_data(num_samples, num_features):
    np.random.seed(42)  # For reproducibility
    X = 2 * np.random.rand(num_samples, num_features)  # Features
    # Labels with some noise, using only the first feature
    y = 4 + 3 * X[:, 0:1] + np.random.randn(num_samples, 1)
    return X, y


# Parameters for the number of samples and features
num_samples = 1000  # Change this value to control the number of samples
num_features = 1  # Change this value to control the number of features

# Generate some example data
X, y = generate_data(num_samples, num_features)

# Initialize parameters (weights and bias)
# Random initial weights [w1, w2, ..., b]
theta = np.random.randn(num_features + 1, 1)
theta_history = [theta]
# Add a bias term (intercept) to the input features
X_b = np.c_[np.ones((num_samples, 1)), X]  # Add x0 = 1 to each instance

# Gradient Descent settings
learning_rate = 0.1
decay_rate = 1
n_iterations = 200
m = len(X_b)

# Directory to save frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Gradient Descent algorithm
for iteration in range(n_iterations):
    # Compute the predictions
    y_pred = X_b.dot(theta)

    # Calculate the error
    error = y_pred - y

    # Compute the gradient
    gradients = 2/m * X_b.T.dot(error)

    # Update the parameters (weights and bias)
    theta = theta - learning_rate * gradients

    # Plot the progress
    plt.scatter(X, y, label='Data points')
    plt.plot(X, y_pred, color='red', label='Fitted line')
    plt.title(f"Iteration {iteration + 1}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    # Save the frame
    plt.savefig(f"frames/frame_{iteration}.png")
    plt.close()

# Create video from frames
frame_array = []
for i in range(n_iterations):
    filename = f'frames/frame_{i}.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter('gradient_descent_progress.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()

# Clean up frames directory (optional)
for file in os.listdir('frames'):
    os.remove(os.path.join('frames', file))
os.rmdir('frames')

print("Video saved as 'gradient_descent_progress.mp4'")
