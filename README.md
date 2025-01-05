## Table of Contents

- [Overview](#overview)
- [Components](#components)
  - [Value](#value)
  - [Neuron](#neuron)
  - [Layer](#layer)
  - [MLP](#mlp)
- [Training Pipeline](#training-pipeline)
- [Usage](#usage)
- [Key Takeaways](#key-takeaways)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Micrograd is a library designed to help users intuitively understand the core principles behind automatic differentiation and neural network training. This implementation recreates its functionality by defining basic building blocks like `Value` objects, neurons, layers, and multi-layer perceptrons (MLPs), enabling gradient computation and backpropagation.

---

## Components

### Value
The `Value` class is the foundational component, representing a scalar value in a computational graph. Each `Value` object tracks its history of operations to enable gradient computation.

**Key Features:**
- Stores a scalar value.
- Tracks the gradient associated with the value.
- Maintains references to parent nodes in the computational graph for backpropagation.

**Example:**
```python
from micrograd import Value

a = Value(2.0)
b = Value(3.0)
c = a * b  # Computational graph tracks this operation
d = c + a

d.backward()  # Compute gradients
print(a.grad)  # Gradient of d w.r.t a
```

---

### Neuron
The `Neuron` class represents a single unit in a neural network. It:
- Computes a weighted sum of inputs.
- Applies a non-linear activation function, like ReLU.

**Example:**
```python
from micrograd import Neuron

n = Neuron(input_size=3)
inputs = [Value(1.0), Value(2.0), Value(-1.0)]
output = n(inputs)  # Forward pass
output.backward()  # Backpropagation
```

---

### Layer
The `Layer` class comprises multiple neurons working together. It:
- Organizes a group of neurons.
- Processes input data collectively.

**Example:**
```python
from micrograd import Layer

layer = Layer(num_inputs=3, num_neurons=4)
inputs = [Value(1.0), Value(2.0), Value(-1.0)]
outputs = layer(inputs)  # Forward pass
```

---

### MLP
The `MLP` (Multi-Layer Perceptron) class stacks multiple layers to create a full neural network. It:
- Accepts input data and propagates it through all layers.
- Outputs predictions based on learned weights and biases.

**Example:**
```python
from micrograd import MLP

mlp = MLP(input_size=3, layer_sizes=[4, 4, 1])
inputs = [Value(1.0), Value(2.0), Value(-1.0)]
output = mlp(inputs)  # Forward pass
output.backward()  # Compute gradients
```

---

## Training Pipeline

1. **Initialize the Model:** Define an MLP with the desired architecture.
2. **Forward Pass:** Pass input data through the MLP to compute predictions.
3. **Compute Loss:** Use a loss function, like mean squared error, to measure prediction accuracy.
4. **Backpropagation:** Call `.backward()` on the loss to compute gradients for all parameters.
5. **Parameter Update:** Adjust parameters using gradient descent.
6. **Repeat:** Iterate over multiple epochs until the model converges.

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/yourusername/micrograd-implementation.git
cd micrograd-implementation
```

2. Import and use the core components in your project. Refer to the examples for guidance.

---

## Key Takeaways

- **Value Object:** Tracks operations and gradients in the computational graph.
- **Neuron:** Encapsulates the weighted sum and activation of inputs.
- **Layer:** Groups neurons for collective input processing.
- **MLP:** Stacks layers to create a complete neural network.

This modular approach simplifies the implementation of backpropagation and gradient-based optimization, making it easier to experiment with small-scale neural networks.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project is inspired by the [micrograd](https://github.com/karpathy/micrograd) library by Andrej Karpathy.
