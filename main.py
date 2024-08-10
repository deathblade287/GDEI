import numpy as np
import matplotlib.pyplot as plt

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

# New efficiency calculation based on Efficiency Index (EI)


def calculate_efficiency(prev_cost, current_cost, prev_grad_norm,
                         current_grad_norm, alpha=1.0, beta=1.0,
                         gamma=0.1, epsilon=1e-8):
    # Calculate the change in loss
    delta_L = prev_cost - current_cost

    # Calculate the Efficiency Index (EI)
    exponential_term = np.exp(-alpha * delta_L / (current_grad_norm + epsilon))
    relative_change = abs((current_cost - prev_cost) / (prev_cost + epsilon))
    quadratic_term = 1 / (1 + beta * (relative_change - gamma) ** 2)

    efficiency = (100 / (1 + exponential_term)) * quadratic_term
    return efficiency

# Function to train the model using gradient descent


def train(X, y, learning_rate, decay_rate, n_iterations):
    # Get the number of samples and features from the input data
    num_samples, num_features = X.shape

    # Initialize weights and bias randomly
    theta = np.random.randn(num_features + 1, 1)

    # Add a bias term (column of ones) to the input features
    X_b = np.c_[np.ones((num_samples, 1)), X]

    # Number of training examples
    m = len(X_b)

    # Initialize lists to store the history of cost and efficiency
    cost_history = []
    efficiency_history = []

    # Variable to store the initial gradient norm
    prev_grad_norm = None

    # Iterate over the specified number of iterations
    for iteration in range(n_iterations):
        # Compute the predicted output based on the current model parameters
        y_pred = X_b.dot(theta)

        # Calculate the error between the predictions and the actual labels
        error = y_pred - y

        # Compute the gradient of the cost function with respect to the parameters
        gradients = 2/m * X_b.T.dot(error)

        # Update the model parameters using the gradients and learning rate
        theta = theta - learning_rate * gradients

        # Calculate the current cost (Mean Squared Error)
        cost = (1/m) * np.sum(error**2)

        # Append the current cost to the history
        cost_history.append(cost)

        # Calculate the norm of the current gradient
        grad_norm = np.linalg.norm(gradients)

        # Skip efficiency calculation for the first iteration
        if iteration == 0:
            prev_grad_norm = grad_norm
            continue

        # Calculate the efficiency metric based on the current and previous costs
        efficiency = calculate_efficiency(
            cost_history[iteration - 1], cost, prev_grad_norm, grad_norm)

        # Append the current efficiency to the history
        efficiency_history.append(efficiency)

        # Print the efficiency for the current iteration
        print(f"Iteration {iteration + 1}: Efficiency = {efficiency}")

        # Update previous gradient norm
        prev_grad_norm = grad_norm

        # Decay the learning rate according to the decay rate
        learning_rate *= decay_rate

    # Return the final model parameters and the efficiency history
    return theta, efficiency_history

# Function to plot efficiency over iterations


def plot_efficiency(efficiency_history, n_iterations):
    # Plot the efficiency history against the number of iterations
    plt.plot(range(1, n_iterations), efficiency_history)

    # Label the x-axis as 'Iterations'
    plt.xlabel('Iterations')

    # Label the y-axis as 'Efficiency'
    plt.ylabel('Efficiency')

    # Set the title of the plot
    plt.title('Efficiency over Iterations')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    num_samples = 100
    num_features = 1
    learning_rate = 0.1
    decay_rate = 0.99
    n_iterations = 50

    # Generate the synthetic data using the specified number of samples and features
    X, y = generate_data(num_samples, num_features)

    # Train the model using gradient descent and obtain the final parameters and efficiency history
    theta, efficiency_history = train(
        X, y, learning_rate, decay_rate, n_iterations)

    # Output the final model parameters (weights and bias)
    print("Final parameters (weights):", theta)

    # Plot the efficiency over the iterations
    plot_efficiency(efficiency_history, n_iterations)
