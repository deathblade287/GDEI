import numpy as np
import matplotlib.pyplot as plt
import math

# Function to generate synthetic data
def generate_data(num_samples, num_features):
    # Set the random seed for reproducibility
    np.random.seed(42)

    # Generate random features in the range [0, 2)
    X = 2 * np.random.rand(num_samples, num_features)

    # Generate labels with a linear relationship and some added noise
    y = 4 + 3 * X[:, 0:1] + np.random.randn(num_samples, 1)

    # Return the generated features and labels
    return X, y

# Function to calculate efficiency
# Function to calculate efficiency with separate variables for each term
def calculate_efficiency(initial_error, cost, prev_cost):
    # Term 1: Error reduction relative to the initial error
    error_reduction = initial_error - cost
    relative_error_reduction = error_reduction / initial_error

    # Term 2: Difference between previous cost and current cost
    cost_difference = abs(prev_cost - cost)

    # Term 3: Logarithmic term to control efficiency
    log_term = 1 + math.log(1 + cost_difference ** 2)

    # Term 4: Combined relative error reduction and logarithmic term
    efficiency_ratio = 100 * relative_error_reduction / log_term

    # Final efficiency calculation, clamped between 1 and 100
    efficiency = 100 - min(100, max(1, efficiency_ratio))

    return efficiency

# Function to perform gradient descent and return efficiency history
def train_model(X, y, learning_rate, decay_rate, n_iterations):
    num_samples, num_features = X.shape
    
    # Initialize parameters (weights and bias)
    theta = np.random.randn(num_features + 1, 1)
    
    # Add a bias term (intercept) to the input features
    X_b = np.c_[np.ones((num_samples, 1)), X]  # Add x0 = 1 to each instance

    # Storage for tracking progress
    cost_history = []
    efficiency_history = []

    # Gradient Descent algorithm
    initial_error = None

    for iteration in range(n_iterations):
        # Compute the predictions
        y_pred = X_b.dot(theta)

        # Calculate the error
        error = y_pred - y

        # Compute the gradient
        gradients = 2/num_samples * X_b.T.dot(error)

        # Update the parameters (weights and bias)
        theta = theta - learning_rate * gradients

        # Calculate the cost (MSE)
        cost = (1/num_samples) * np.sum(error**2)
        cost_history.append(cost)

        # Set the initial error for the first iteration
        if iteration == 0:
            initial_error = cost
            continue  # Skip efficiency calculation for the first iteration

        # Calculate the efficiency
        prev_cost = cost_history[iteration - 1]
        efficiency = calculate_efficiency(initial_error, cost, prev_cost)
        efficiency_history.append(efficiency)
        print(f"Iteration {iteration + 1}: Efficiency = {efficiency}")

        # Decay the learning rate
        learning_rate *= decay_rate

    return theta, efficiency_history

# Function to plot efficiency over iterations
def plot_efficiency(efficiency_history, n_iterations):
    plt.plot(range(1, n_iterations), efficiency_history)
    plt.xlabel('Iterations')
    plt.ylabel('Efficiency')
    plt.show()

# Execute the code
if __name__ == "__main__":
    # Parameters for the number of samples and features
    num_samples = 100
    num_features = 1

    # Generate data
    X, y = generate_data(num_samples, num_features)

    # Gradient Descent settings
    learning_rate = 0.1
    decay_rate = 0.7
    n_iterations = 15

    # Train the model and get the efficiency history
    theta, efficiency_history = train_model(X, y, learning_rate, decay_rate, n_iterations)

    # Output the final parameters
    print("Final parameters (weights):", theta)

    # Plot the efficiency
    plot_efficiency(efficiency_history, n_iterations)