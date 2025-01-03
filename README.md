# Micrograd Clone

This repository is a clone of the micrograd project, which is a minimalistic implementation of a neural network library. The project is designed to help understand the core concepts of neural networks and backpropagation by building them from scratch.

## Overview

The project includes a series of Jupyter notebooks that walk through the implementation of a simple neural network library. The library supports basic operations such as addition, multiplication, and the tanh activation function, and it includes functionality for automatic differentiation via backpropagation.

## Features

- **Value Class**: Represents a scalar value and supports operations like addition and multiplication.
- **Automatic Differentiation**: Implements backpropagation to compute gradients.
- **Graph Visualization**: Uses Graphviz to visualize the computation graph of expressions.
- **Neural Network Example**: Demonstrates a simple neuron with two inputs and a bias.

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required Python packages: `numpy`, `matplotlib`, `graphviz`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/micrograd_clone.git
   cd micrograd_clone
   ```

2. Install the required packages:
   ```bash
   pip install numpy matplotlib graphviz
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open `micrograd_clone.ipynb` in Jupyter Notebook to explore the code and run the examples.

## Usage

The main file `micrograd_clone.ipynb` contains the implementation and examples. You can run the notebook cells to see how the library is built and how it can be used to perform forward and backward passes through a simple neural network.

## Code Structure

- **Value Class**: Implements the core functionality for scalar values, including operations and gradient tracking.
- **Backpropagation**: The `backward` method in the `Value` class computes gradients using a topological sort of the computation graph.
- **Visualization**: The `draw_dot` function uses Graphviz to create a visual representation of the computation graph.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by the [micrograd](https://github.com/karpathy/micrograd) library by Andrej Karpathy.
