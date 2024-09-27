import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [64]
NUM_OUTPUT = 10
REG_CONST = 0.5


def unpack(weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT * NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i] * NUM_HIDDEN[i + 1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1] * NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i - 1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i + 1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs


def relu_grad(z):
    return (z > 0).astype(float)


def fCE(X, Y, weights):
    Ws, bs = unpack(weights)

    h = X
    for i in range(NUM_HIDDEN_LAYERS - 1):
        z = np.dot(h, Ws[i].T) + bs[i]
        h = relu(z)
    # softmax
    predicted_label = calc_z_and_softmax(h, Ws[-1], bs[-1])
    ce_loss = ((-np.sum(Y * np.log(predicted_label + 1e-10)) / Y.shape[0]) +
               ((REG_CONST / 2) * np.sum(np.dot(Ws[-1].T, Ws[-1]))))

    reg_term = (REG_CONST / 2) * np.sum([np.sum(W ** 2) for W in Ws])

    ce_loss = ce_loss + reg_term
    return ce_loss


def calc_z_and_softmax(features, w, b):
    # compute z
    z = np.dot(features, w.T) + b
    # softmax
    exp_x = np.exp(z - np.max(z, axis=1, keepdims=True))
    predicted_label = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return predicted_label


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def gradCE(X, Y, weights):
    Ws, bs = unpack(weights)
    grads_Ws = [None] * len(Ws)  # Initialize gradients for weights
    grads_bs = [None] * len(bs)  # Initialize gradients for biases

    # Forward pass to compute activations
    h = X
    activations = [h]  # Store activations for backprop
    for i in range(NUM_HIDDEN_LAYERS - 1):
        print(f"Shape of h: {h.shape}, Shape of Ws[{i}]: {Ws[i].shape}")
        z = np.dot(h, Ws[i].T) + bs[i]
        h = relu(z)
        activations.append(h)

    # Compute softmax and loss (not used in backprop, but useful to keep track)
    predicted_label = calc_z_and_softmax(h, Ws[-1], bs[-1])

    # Backward pass (backpropagation)
    # Compute gradient at output layer (softmax layer)
    delta = predicted_label - Y  # Gradient of cross-entropy loss with respect to softmax output
    grads_Ws[-1] = np.dot(activations[-2].T, delta) / X.shape[0] + REG_CONST * Ws[-1]

    grads_bs[-1] = np.sum(delta, axis=0) / X.shape[0]  # Gradient of biases (no regularization)

    # Backpropagate through hidden layers
    for i in range(NUM_HIDDEN_LAYERS - 2, -1, -1):
        delta = np.dot(delta, Ws[i + 1].T) * relu_derivative(activations[i + 1])  # Backprop delta
        grads_Ws[i] = np.dot(activations[i].T, delta) / X.shape[0] + REG_CONST * Ws[
            i]
        grads_bs[i] = np.sum(delta, axis=0) / X.shape[0]  # Gradient of biases

    # Pack gradients and return
    gradients = pack(grads_Ws, grads_bs)
    return gradients


def pack(grads_Ws, grads_bs):
    # Flatten and concatenate all gradient matrices and bias vectors into a single 1D array
    return np.hstack([W.flatten() for W in grads_Ws] + [b.flatten() for b in grads_bs])


# Creates an image representing the first layer of weights (W0).
def show_W0(W):
    Ws, bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([np.pad(np.reshape(W[idx1 * n + idx2, :], [28, 28]), 2, mode='constant') for idx2 in range(n)]) for
        idx1 in range(n)
    ]), cmap='gray'), plt.show()


def initWeightsAndBiases():
    Ws = []
    bs = []

    # Strategy:
    # Sample each weight using a variant of the Kaiming He Uniform technique.
    # Initialize biases to small positive number (0.01).

    np.random.seed(0)
    W = 2 * (np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN[i + 1], NUM_HIDDEN[i])) / NUM_HIDDEN[i] ** 0.5) - 1. / NUM_HIDDEN[
            i] ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i + 1])
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1])) / NUM_HIDDEN[-1] ** 0.5) - 1. / NUM_HIDDEN[-1] ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs


def train(trainX, trainY, weights, testX, testY, lr=5e-2, num_epochs=100, batch_size=32):
    # Initialize variables to store loss history
    train_loss_history = []
    test_loss_history = []
    n_samples = trainX.shape[0]

    # Main training loop
    for epoch in range(num_epochs):
        # Shuffle data for stochastic gradient descent
        indices = np.random.permutation(n_samples)
        trainX_shuffled = trainX[indices]
        trainY_shuffled = trainY[indices]

        for i in range(0, n_samples, batch_size):
            # Create mini-batches
            X_batch = trainX_shuffled[i:i + batch_size]
            Y_batch = trainY_shuffled[i:i + batch_size]

            # 1. Compute the gradients using backpropagation with regularization on the mini-batch
            gradients = gradCE(X_batch, Y_batch, weights)

            # 2. Update the weights using gradient descent on the mini-batch
            weights = weights - lr * gradients

        # 3. Compute the cost (cross-entropy loss + regularization) for the current weights
        train_loss = fCE(trainX, trainY, weights)
        test_loss = fCE(testX, testY, weights)

        # 4. Store the loss values for visualization and monitoring
        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)

        # 5. Print the progress (optional)
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    # Return the final weights and the loss history for both training and test sets
    return weights, train_loss_history, test_loss_history

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

if __name__ == "__main__":
    # Load training data.
    Ws, bs = initWeightsAndBiases()
    trainX = np.load("fashion_mnist_train_images.npy")
    trainY = np.load("fashion_mnist_train_labels.npy")
    testX = np.load("fashion_mnist_test_images.npy")
    testY = np.load("fashion_mnist_test_labels.npy")

    trainY = one_hot_encode(trainY, NUM_OUTPUT)
    testY = one_hot_encode(testY, NUM_OUTPUT)

    print(f"trainY shape: {trainY.shape}")  # Should print (60000, 10)
    print(f"testY shape: {testY.shape}")
    print(trainY.shape)
    # Pack all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

    # On just the first 5 training examples, do numeric gradient check.
    print(scipy.optimize.check_grad(
        lambda weights_: fCE(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[:5]), weights_),
        lambda weights_: gradCE(np.atleast_2d(trainX[:5]), np.atleast_2d(trainY[:5]), weights_),
        weights))

    # Train with stochastic gradient descent
    final_weights, train_loss_history, test_loss_history = train(trainX, trainY, weights, testX, testY, lr=0.01,
                                                                 batch_size=32)

    # Visualize the first layer of weights
    show_W0(final_weights)
