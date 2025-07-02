from PIL import Image
import numpy as np

from helpers.cnn import convolutional

def load_input(path):
    img = Image.open(path).convert("L")
    arr = np.array(img) / 255.0
    return arr

def main():
    fname = input("What character photo would you like to test? ")
    input_val = load_input("characters/" + fname + ".png")
    y_test = input("What is the expected result? ")

    index_to_char = {i: char for i, char in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}

    weights, biases, kernels = load_model(0, 16)

    feature_maps, _, _ = convolutional(input_val, kernels)

    pooled = np.array(feature_maps)

    flatten = pooled.flatten()

    # Predict using feedforward, but only interested in probabilities
    logits = np.dot(flatten, weights) + biases
    y_pred_probs = softmax(logits)
    y_pred_class = np.argmax(y_pred_probs)

    print(f"\nActual Value = {y_test}")
    print(f"CNN Output: {ord(index_to_char[y_pred_class])}")
    print(f"ASCII -> Character: {index_to_char[y_pred_class]}")

def softmax(x):
    exps = np.exp(x - np.max(x))  # for numerical stability
    return exps / np.sum(exps)

def load_model(pooling, kernel=3):
    filename = "model/" + str(kernel) + "/"
    if pooling == 1:
        filename += "ocr_model_mean_pooling.npz"
    elif pooling == 2:
        filename += "ocr_model_max_pooling.npz"
    else:
        filename += "ocr_model_no_pooling.npz"

    data = np.load(filename, allow_pickle=True)
    weights = data["weights"]
    biases = data["biases"]
    kernels = data["kernels"]
    return weights, biases, kernels

if __name__ == "__main__":
    main()