import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file containing Study_Time and Score
data = pd.read_csv('data.csv')  

# Define the loss function (Mean Squared Error)
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        # Get x and y values from the dataset
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score
        # Calculate squared error for this point
        total_error += (y - (m * x + b)) ** 2
    # Return mean squared error
    return total_error / float(len(points))

# Define gradient descent to update m (slope) and b (intercept)
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0  # gradient for slope
    b_gradient = 0  # gradient for intercept
    n = len(points)  # number of data points

    for i in range(n):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score

        # Partial derivative with respect to m (slope)
        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        # Partial derivative with respect to b (intercept)
        b_gradient += (-2/n) * (y - (m_now * x + b_now))

    # Update m and b using learning rate L
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Initialize parameters
m = 0  # initial slope
b = 0  # initial intercept
L = 0.0001  # learning rate
epochs = 300  # number of iterations

# Run gradient descent
for i in range(epochs):
    # Print loss every 50 epochs to track progress
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss_function(m, b, data):.4f}")
    m, b = gradient_descent(m, b, data, L)

# Print final slope and intercept
print(f"Final m: {m}, b: {b}")

# Plot the data points
plt.scatter(data.Study_Time, data.Score, color="black", label="Data points")

# Plot the regression line
x_vals = np.linspace(data.Study_Time.min(), data.Study_Time.max(), 100)
y_vals = m * x_vals + b
plt.plot(x_vals, y_vals, color="red", label="Regression line")

# Add labels and legend
plt.xlabel("Study Time")
plt.ylabel("Score")
plt.legend()
plt.show()
