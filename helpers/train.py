import numpy as np

from helpers.cnn import convolutional, mean_pooling, max_pooling, backpropagation_cnn
from helpers.ann import initialize_nn, feedforward, backpropagation

def train(epochs, pooling=0):
    train_data = np.load('dataset/train_dataset.npz')
    X_train, y_train = train_data['X'], train_data['y']

    total_inputs = 15
    total_outputs = 10
    learning_rate = 0.005
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    kernels = []
    for i in range(1):
        kernels.append(np.random.randn(3, 3) * 0.1)

    if pooling in [1, 2]:
        total_inputs = 8

    weights, biases = initialize_nn(total_inputs, total_outputs)

    for epoch in range(epochs):
        epoch_loses = []
        # Parse through all training data
        for i in range(1000):
            feature_maps, patches, relu_mask = convolutional(X_train[i], kernels)
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

            # ANN Feed Forward
            y_onehot = to_one_hot(y_train[i])
            _, gradient, loss = feedforward(flatten, y_onehot, weights, biases)
            epoch_loses.append(loss)

            # ANN Backpropagation
            backpropagation(flatten, gradient, weights, biases, learning_rate)

            # CNN Backpropagation
            d_flatten = np.dot(gradient, weights.T)
            grad_from_next = d_flatten.reshape(pooled.shape)

            kernels = backpropagation_cnn(grad_from_next, patches, relu_mask, kernels, learning_rate)

        mean_loss = np.mean(epoch_loses)
        print(f"Epoch {epoch} - Loss: {mean_loss:.6f}")

        if mean_loss < best_loss:
            best_loss = mean_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping. No improvements!")
                break
    
    save_model(weights, biases, kernels, pooling)


def to_one_hot(label_index, num_classes=10):
    one_hot = np.zeros(num_classes)
    one_hot[label_index] = 1
    return one_hot


def save_model(weights, biases, kernels, pooling):
    filename = "ocr_model_no_pooling.npz"

    if pooling == 1:
        filename = "ocr_model_mean_pooling.npz"
    elif pooling == 2:
        filename = "ocr_model_max_pooling.npz"
    np.savez(filename, weights=weights, biases=biases, kernels=kernels)


if __name__ == "__main__":
    train(epochs=100)