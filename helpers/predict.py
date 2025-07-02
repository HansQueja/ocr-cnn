import numpy as np

from helpers.cnn import convolutional, mean_pooling, max_pooling

def predict(pooling=0, kernel=3):
    train_data = np.load('dataset/alphanumeric_test_dataset.npz')
    X_test, y_test = train_data['X'], train_data['y']

    index_to_char = {i: char for i, char in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}

    weights, biases, kernels = load_model(pooling, kernel)

    correct = 0

    for i in range(len(y_test)):

        feature_maps, _, _ = convolutional(X_test[i], kernels)
        pooled = None

        # If mean pooling
        if pooling == 1:
            #print("Performing Mean Pooling")
            pooled = mean_pooling(feature_maps, pool_size=(2, 2), stride=(1, 1))
        elif pooling == 2:
            #print("Performing Max Pooling")
            pooled = max_pooling(feature_maps, pool_size=(2, 2), stride=(1, 1))
        else:
            #print("Performing No Pooling")
            pooled = np.array(feature_maps)

        flatten = pooled.flatten()

        # Predict using feedforward, but only interested in probabilities
        logits = np.dot(flatten, weights) + biases
        y_pred_probs = softmax(logits)
        y_pred_class = np.argmax(y_pred_probs)

        print(f"Sample {i}: Predicted = {index_to_char[y_pred_class]}, Actual = {index_to_char[y_test[i]]}")

        if y_pred_class == y_test[i]:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(X_test)} = {correct / len(X_test):.3%}")

def to_one_hot(label_index, num_classes=10):
    one_hot = np.zeros(num_classes)
    one_hot[label_index] = 1
    return one_hot

def softmax(x):
    exps = np.exp(x - np.max(x))  # for numerical stability
    return exps / np.sum(exps)

def load_model(pooling, kernel):
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
    predict(pooling=0)