import numpy as np
import matplotlib.pyplot as plt

# Define the activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def linear(x):
    return x

def relu(x):
    return np.maximum(0, x)

def binary_step(x):
    return np.where(x >= 0, 1, 0)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate a range of x values
x = np.linspace(-10, 10, 400)

# Compute the outputs for each activation function
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_linear = linear(x)
y_relu = relu(x)
y_binary_step = binary_step(x)
y_leaky_relu = leaky_relu(x)

# Plot each activation function
plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.plot(x, y_sigmoid)
plt.title('Sigmoid Function')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(x, y_tanh)
plt.title('Hyperbolic Tangent Function')
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(x, y_linear)
plt.title('Linear Function')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(x, y_relu)
plt.title('ReLU Function')
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(x, y_binary_step)
plt.title('Binary Step Function')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(x, y_leaky_relu)
plt.title('Leaky ReLU Function')
plt.grid(True)

plt.tight_layout()
plt.show()

