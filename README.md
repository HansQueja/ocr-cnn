# Optical Character Recognition: Raw CNN Implementation from Scratch

This project is a low-level implementation of a **Convolutional Neural Network (CNN)** built entirely from the ground up using only **NumPy**. This is my raw dive towards the intricacies of CNN, where I also tried to implement the mathematical theories behind the gradient descent and even the convolutions.

## Technical Specifications

* **Input Dimension**:  pixel grayscale inputs representing handwritten characters (A–Z and 0–9).
* **Kernel Operations**: Supports varying kernel counts (1, 3, 8, 16) to explore feature extraction depth.
* **Pooling Layers**: Custom-built max-pooling logic with options for different stride/pooling configurations to achieve spatial invariance.
* **Data Pipeline**: Manually designed training pipeline with ASCII code mapping and custom activation functions.

## Mathematical Foundation

The model updates its weights via a manually implemented gradient descent. For a given loss function, the weight update for a kernel  is calculated as:

Where  represents the learning rate and  is the partial derivative of the loss with respect to the kernel weights, computed via the chain rule across the pooling and convolutional layers.

## Usage and Simulation

The system is designed with a command-line interface (CLI) to allow for rapid experimentation with different hyperparameters.

### Command Template

```bash
python main.py --method [train, predict] --epochs [integer] --pooling [0-2] --kernel [1,3,8,16]

```

### Examples

* **To Train**: Execute 1,000 epochs with pooling enabled to optimize feature reduction.
```bash
python main.py --method train --epochs 1000 --pooling 1

```


* **To Predict**: Test the model's inference using 8 kernels without a pooling layer.
```bash
python main.py --method predict --pooling 0 --kernel 8

```
