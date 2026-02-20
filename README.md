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

## Validation and Verification

<img width="887" height="570" alt="image" src="https://github.com/user-attachments/assets/adf8049f-0196-4bd8-ac0c-c81674789647" />

The training loss graph shows the model successfully learned the OCR patterns. Starting at 4.1, the loss steadily decreased to approximately 0.4 by epoch 400. The smooth curve, free of fluctuations, indicates stable training with good convergence and no overfitting.

<img width="940" height="574" alt="image" src="https://github.com/user-attachments/assets/35e2a700-bfb8-416d-9e53-7dd242bf91a3" />

The graph visualizes the model's learning progression. Starting near 0% accuracy, the model steadily improved in a stepwise pattern, reaching approximately 40% by epoch 100, then continued to climb through several accuracy plateaus. By epoch 250, the accuracy achieved around 95% and stabilized near 100% by epoch 300, maintaining this high performance through epoch 400.

## CNN Training

<img width="882" height="573" alt="image" src="https://github.com/user-attachments/assets/2f882324-8a00-45f4-a20a-3dfbf6c4ede7" />

*OCR-CNN Predicting Accuracy: Max Pooling - 5 Kernels*

<img width="872" height="570" alt="image" src="https://github.com/user-attachments/assets/4740952c-d9b4-41f4-ad75-e0a9a8b946a0" />

*OCR-CNN Predicting Accuracy: Mean Pooling - 5 Kernels*

<img width="869" height="442" alt="image" src="https://github.com/user-attachments/assets/91eaa9ab-ba7d-489f-989b-e5cf26281969" />

*OCR-CNN Predicting Accuracy: No Pooling - 5 Kernels*

<img width="863" height="557" alt="image" src="https://github.com/user-attachments/assets/dece5a12-f828-4f0f-9f0f-b43b467bbe3d" />

*OCR-CNN Predicting Accuracy: No Pooling - 8 Kernels*


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
