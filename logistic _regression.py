import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ===============================
# Logistic Regression Functions
# ===============================

def sigmoid(z):
    """This is the sigmoid function. It squashes numbers between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradient(theta, X_b, y):
    """
    Compute the gradient of the loss function.
    The gradient tells us the direction to move our weights
    so that our predictions get closer to the real values.
    """
    m = y.size
    return (X_b.T @ (sigmoid(X_b @ theta) - y)) / m

def gradient_descent(X, y, alpha=0.1, num_iter=100, tol=1e-7):
    """
    Learn the weights (theta) using gradient descent.
    - alpha: how big of a step to take each iteration
    - num_iter: max number of steps
    - tol: if the gradient is tiny, stop early
    """
    # Add a column of ones for the bias term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])  # start with all zeros

    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad  # move in the direction that reduces loss

        if np.linalg.norm(grad) < tol:
            break  # if the change is tiny, we’re done

    return theta

def predict_proba(X, theta):
    """Return predicted probabilities for each sample."""
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias column
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold=0.5):
    """Turn probabilities into 0 or 1 based on a threshold."""
    return (predict_proba(X, theta) >= threshold).astype(int)


# ===============================
# Load and Prepare Data
# ===============================

# Load the breast cancer dataset
X, y = load_breast_cancer(return_X_y=True)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the features so all of them are on a similar scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# Train the Model
# ===============================

# Use gradient descent to learn theta
theta_hat = gradient_descent(X_train_scaled, y_train, alpha=0.1)

# Make predictions
y_pred_train = predict(X_train_scaled, theta_hat)
y_pred_test = predict(X_test_scaled, theta_hat)

# Check accuracy
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)


# ===============================
# Visualize Predictions in 2D
# ===============================

# Since the data has 30 features, we reduce it to 2 using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

plt.figure(figsize=(10,5))

# Plot training points (colored by their true labels)
plt.scatter(
    X_train_pca[:,0], X_train_pca[:,1], 
    c=y_train, cmap='coolwarm', alpha=0.6, label='Train Labels'
)

# Plot testing points predictions (crosses)
plt.scatter(
    X_test_pca[:,0], X_test_pca[:,1], 
    c=y_pred_test, cmap='coolwarm', marker='x', label='Test Predictions'
)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Breast Cancer Dataset: True Labels vs Predicted")
plt.legend()
plt.show()
