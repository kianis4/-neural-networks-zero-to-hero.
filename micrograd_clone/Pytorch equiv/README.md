# Micrograd Clone with PyTorch

This repository demonstrates how the functionality of [Micrograd](https://github.com/karpathy/micrograd), a lightweight library for autograd and neural network implementation, can be recreated using PyTorch's tensor and autograd capabilities. The project is presented in a step-by-step, intuitive manner to help you understand the concepts and implementation thoroughly.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Implementation Details](#implementation-details)
  - [Core Concepts](#core-concepts)
  - [Steps to Recreate Micrograd Functionality](#steps-to-recreate-micrograd-functionality)
- [Usage](#usage)
- [Examples](#examples)
  - [Manual Gradient Calculation](#manual-gradient-calculation)
  - [Neural Network Example](#neural-network-example)
- [Key Takeaways](#key-takeaways)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Micrograd is a simple and powerful library built to understand the underlying principles of automatic differentiation and neural networks. In this project, we illustrate how to achieve similar functionality using PyTorch, leveraging its `torch.Tensor` and autograd capabilities. PyTorch's tools provide a robust foundation to explore machine learning concepts while maintaining flexibility and ease of use.

---

## Installation

Ensure you have Python 3.8 or higher installed along with PyTorch. Install the required dependencies by running:

```bash
pip install torch
```

To clone this repository:

```bash
git clone https://github.com/yourusername/micrograd-clone-pytorch.git
cd micrograd-clone-pytorch
```

---

## Implementation Details

### Core Concepts

Micrograd operates on the following principles:

1. **Scalar-Based Computation:** It performs operations on individual values while tracking the computational graph.
2. **Automatic Differentiation:** Uses reverse-mode differentiation to compute gradients.
3. **Neural Network Representation:** Defines simple layers and activation functions for building and training small models.

Using PyTorch, we achieve:

- **Tensor-Based Computation:** Extends scalar operations to multidimensional arrays.
- **Efficient Autograd:** Seamlessly computes gradients for tensors with respect to any computation graph.
- **Scalable Neural Networks:** Easily builds networks using `torch.nn`.

### Steps to Recreate Micrograd Functionality

1. **Scalar Representation:** Use `torch.Tensor` objects with `requires_grad=True` to mimic the `Value` object in Micrograd.
2. **Operations and Gradients:** Leverage PyTorch's built-in support for gradient computation during operations.
3. **Manual Backpropagation:** Use `.backward()` for gradient propagation similar to Micrograd's custom backprop logic.
4. **Building Neural Networks:** Implement basic layers and activation functions using `torch.nn` to show PyTorch's versatility.

---

## Usage

1. Open the `micrograd_clone_pytorch.ipynb` notebook.
2. Follow the step-by-step implementation to recreate Micrograd's functionality with PyTorch.
3. Experiment with examples to deepen your understanding.

---

## Examples

### Manual Gradient Calculation

Below is an example of a simple scalar operation:

```python
import torch

# Define scalars with requires_grad=True
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

# Perform operations
c = a * b
d = c + a

# Backpropagate
d.backward()

print(a.grad)  # Gradient of d w.r.t a
print(b.grad)  # Gradient of d w.r.t b
```

### Neural Network Example

This example shows how to build and train a simple neural network:

```python
import torch
import torch.nn as nn

# Define a simple model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model
model = SimpleNN()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Dummy data
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
targets = torch.tensor([[1.0], [0.0]])

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

print("Training complete!")
```

---

## Key Takeaways

- PyTorch provides an efficient way to recreate Micrograd functionality with extended support for tensor operations and autograd.
- Gradient tracking and computational graph creation are implicit in PyTorch, making it more user-friendly for large-scale tasks.
- The neural network abstraction in PyTorch (`torch.nn`) enables building scalable models effortlessly.

---

## Contributing

We welcome contributions to improve the clarity, examples, or functionality of this project. Please submit a pull request or open an issue for suggestions.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.
