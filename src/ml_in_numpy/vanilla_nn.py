import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

# Get MNIST data
DATA_PATH = '../../data/mnist'

# Architecture parameters
NUM_INPUT_NEURONS = 748
NUM_HIDDEN_NEURONS_LIST = [100, 50]
NUM_OUTPUT_NEURONS = 10

# Data parameters
VALIDATION_FRAC = 0.3

# Training parameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.99
MOMENTUM = 0.98
REGULARISATION = 1e-5


class VanillaNN(object):
    def __init__(self, num_in=None, num_hidden=None, num_out=None):
        # Overwrite default architecture parameters if given
        if num_in is not None:
            assert isinstance(num_in, int)
            self.num_in = num_in
        else:
            self.num_in = NUM_INPUT_NEURONS

        if num_hidden is not None:
            if isinstance(num_hidden, int):
                num_hidden = [num_hidden]
            assert isinstance(num_hidden, list)
            self.num_hidden = num_hidden
        else:
            self.num_hidden = NUM_HIDDEN_NEURONS_LIST

        if num_out is not None:
            assert isinstance(num_out, int)
            self.num_out = num_out
        else:
            self.num_out = NUM_OUTPUT_NEURONS

        # Set default data parameters
        self.data_path = DATA_PATH
        self.val_frac = VALIDATION_FRAC

        # Set default training parameters
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.rate = LEARNING_RATE
        self.decay = LEARNING_RATE_DECAY
        self.momentum = MOMENTUM
        self.reg = REGULARISATION

        # Print network architecture
        print('\n # # # Network architecture # # # ')
        print('- input: {:>5}'.format(num_in))
        for i, num in enumerate(num_hidden):
            print('- hidden{}: {}'.format(i + 1, num))
        print('- output: {:>3}\n'.format(num_out))

    def load_data(self, data_path=None, val_frac=None, testing=False):
        # Overwrite defaults if given
        if data_path is not None:
            self.data_path = data_path
        if val_frac is not None:
            self.val_frac = val_frac

        # Data location
        mndata = MNIST(self.data_path)

        if testing:  # Create small datasets from mnist testing data
            X_use, y_use = mndata.load_testing()
            samps = len(X_use)
            perm = np.random.permutation(samps)
            X_use, y_use = np.array(X_use)[perm, :], np.array(y_use)[perm]
            X_temp = X_use[(samps // 4):, :]
            y_temp = y_use[(samps // 4):]
            self.X_test = X_use[:(samps // 4), :]
            self.y_test = y_use[:(samps // 4)]
        else:  # Use all available data
            print('Loading training data...')
            X_temp, y_temp = mndata.load_training()
            print('Loading testing data...')
            self.X_test, self.y_test = mndata.load_testing()

        print('Preparing data for training...\n')
        # Shape input to (samples x features) and normalise
        X_mean, X_max, X_min = np.mean(X_temp), np.max(X_temp), np.min(X_temp)
        X_temp = (np.array(X_temp) - X_mean) / (X_max - X_min)
        self.X_test = (np.array(self.X_test) - X_mean) / (X_max - X_min)

        # One-hot the labels
        y_temp = np.eye(self.num_out)[np.array(y_temp)]
        self.y_test = np.eye(self.num_out)[np.array(self.y_test)]

        # Split temp into training/validation
        perm = np.random.permutation(X_temp.shape[0])
        X_temp = X_temp[perm, :]
        y_temp = y_temp[perm, :]
        val_ind = int(self.val_frac * X_temp.shape[0])
        self.X_train = X_temp[val_ind:, :]
        self.y_train = y_temp[val_ind:, :]
        self.X_val = X_temp[:val_ind, :]
        self.y_val = y_temp[:val_ind, :]

    def train(self, num_epochs=None, batch_size=None, learning_rate=None,
              learning_rate_decay=None, momentum=None, regularisation=None,
              do_momentum=True, learning_curve=True):
        # Overwrite default training parameters if given
        if num_epochs is not None:
            self.num_epochs = num_epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.rate = learning_rate
        if learning_rate_decay is not None:
            self.decay = learning_rate_decay
        if momentum is not None:
            if not do_momentum:
                print('WARNING: momentum parameter specified, but do_momentum '
                      'set to False! Set do_momentum to True if you want to '
                      'use momentum.\n')
            self.momentum = momentum
        if regularisation is not None:
            self.reg = regularisation

        # Initialise weights
        self._init_weights()

        # Initialise for learning curves
        self.train_losses, self.val_losses = [], []

        # Run over the dataset num_epochs times
        for i in range(self.num_epochs):
            print('Iteration {}/{}'.format(i + 1, self.num_epochs))
            # Compute full train and val losses every epoch
            if learning_curve:
                self.train_losses.append(self._Xy_loss(self.X_train,
                                                       self.y_train))
                self.val_losses.append(self._Xy_loss(self.X_val,
                                                     self.y_val))

            # Get mini-batches
            batch_inds = range(0, self.X_train.shape[0], self.batch_size)
            # Mini-batch SGD
            for k in batch_inds:
                X_batch = self.X_train[k:k + self.batch_size, :]
                y_batch = self.y_train[k:k + self.batch_size, :]
                self._forward(X_batch)
                w_updates, b_updates = self._backward(y_batch)

                # Momentum update or normal update
                if do_momentum:
                    self._momentum_update(w_updates, b_updates)
                else:
                    self._update(w_updates, b_updates)

            # Decay learning rate
            self.rate *= self.decay

    def accuracy(self):
        train_acc = self._Xy_accuracy(self.X_train, self.y_train)
        val_acc = self._Xy_accuracy(self.X_val, self.y_val)
        test_acc = self._Xy_accuracy(self.X_test, self.y_test)
        return train_acc, val_acc, test_acc

    def _init_weights(self):
        """
        Xavier initialisation of weights: Gaussian with mean 0, and variance
        (2 / (N_in + N_out)) for every layer.
        """
        self.weights = []
        self.biases = []

        # Input layer
        w_in = np.random.normal(loc=0.0, scale=(2 / np.sqrt(
            (self.num_in + self.num_hidden[0]))), size=(
                self.num_in, self.num_hidden[0]))
        self.weights.append(w_in)
        self.biases.append(np.zeros(self.num_hidden[0]).reshape(1, -1))

        for i in range(len(self.num_hidden)):
            # Output layer
            if i + 1 == len(self.num_hidden):
                w_out = np.random.normal(loc=0.0, scale=(2 / np.sqrt(
                    (self.num_hidden[-1] + self.num_out))), size=(
                        self.num_hidden[-1], self.num_out))
                self.weights.append(w_out)
                self.biases.append(np.zeros(self.num_out).reshape(1, -1))
                break
            # Hidden layers
            w = np.random.normal(loc=0.0, scale=(2 / np.sqrt(
                (self.num_hidden[i] + self.num_hidden[i + 1]))), size=(
                    self.num_hidden[i], self.num_hidden[i + 1]))
            self.weights.append(w)
            self.biases.append(np.zeros(self.num_hidden[i + 1]).reshape(1, -1))

        # Accumulated momentum
        self.w_mom = [np.zeros_like(weight) for weight in self.weights]
        self.b_mom = [np.zeros_like(bias).reshape(1, -1) for bias in
                      self.biases]

    def _forward(self, X_batch):
        # Add input as first z value
        self.z = [X_batch]
        self.act = []

        # print('input', self.z[0][0, :])
        for in_val, weight, bias in zip(self.z, self.weights, self.biases):
            z = np.dot(in_val, weight) + bias
            act = self._relu(z)
            self.z.append(z)
            self.act.append(act)

        # Softmax layer
        self.act.append(self._softmax(act))

    def _backward(self, y_batch):
        w_updates, b_updates = [], []
        deltas = []

        # Output layer has different calculation for delta
        delta_out = (self.act[-1] - y_batch) * self._theta(self.z[-1])
        deltas.append(delta_out)

        # Store weight updates
        w_update = (np.dot(self.z[-2].T, delta_out) + self.reg *
                    self.weights[-1]) / y_batch.shape[0]
        b_update = np.mean(delta_out, axis=0).reshape(1, -1)
        w_updates.append(w_update)
        b_updates.append(b_update)

        # Loop through other layers
        for i in range(len(self.weights) - 1):
            # Same delta calculation every layer from now on
            delta = np.dot(deltas[i], self.weights[-1 - i].T) * self._theta(
                self.z[-2 - i])
            deltas.append(delta)

            # Store weight updates
            w_update = (np.dot(self.z[-3 - i].T, delta) + self.reg *
                        self.weights[-2 - i]) / y_batch.shape[0]
            b_update = np.mean(delta, axis=0).reshape(1, -1)
            w_updates.append(w_update)
            b_updates.append(b_update)

        # Reverse updates so the order corresponds to that in the weights
        w_updates = w_updates[::-1]
        b_updates = b_updates[::-1]
        return w_updates, b_updates

    def _grad_check(self):
        # NOTE: May fail for some random values because of kinks in ReLU
        # TODO: For some reason only the backprop gradients in the final layer
        # correspond to the numerical gradients. Since the network seems to
        # do just fine, might be a gradient check implementation error. In
        # particular, maybe the weights are changed in the background?

        # Number of weights to check in every layer
        num_to_check = 10
        # Get small test set
        inds = np.random.randint(self.X_train.shape[0], size=100)
        X = self.X_train[inds, :]
        y = self.y_train[inds, :]
        # Do initial _forward and _backward to get backprop gradients
        _ = self._forward(X)
        w_updates, b_updates = self._backward(y)
        # Check weights
        print('\nRelative weight error')
        self._num_grad(num_to_check, X, y, self.weights, w_updates)
        # Check biases
        print('\nRelative bias error')
        self._num_grad(num_to_check, X, y, self.biases, b_updates)

    def _num_grad(self, num_to_check, X, y, weights, updates):
        # How much to shift weights
        eps = 1e-5
        # Test num_to_check weights for every layer
        for _ in range(num_to_check):
            print()
            # Loop over weight layers
            for k, weight in enumerate(weights):
                # Pick random weight to shift
                i = np.random.randint(weight.shape[0])
                j = np.random.randint(weight.shape[1])
                # loss(weights + eps)
                weight[i, j] += eps
                lp = self._Xy_loss(X, y)
                # loss(weights - eps)
                weight[i, j] -= 2 * eps
                lm = self._Xy_loss(X, y)
                # set weight back to original value
                weight[i, j] += eps
                # Numerical gradient
                num_grad = (lp - lm) / (2 * eps)
                bp_grad = updates[k][i, j]
                print(abs(num_grad - bp_grad) / max(num_grad, bp_grad))

    def _update(self, w_updates, b_updates):
        for i, w_update, b_update in zip(range(len(w_updates)), w_updates,
                                         b_updates):
            # Update weights
            self.weights[i] -= self.rate * w_update
            self.biases[i] -= self.rate * b_update

    def _momentum_update(self, w_updates, b_updates):
        for i, w_update, b_update in zip(range(len(w_updates)), w_updates,
                                         b_updates):
            # Update accumulated momentum
            self.w_mom[i] = (self.momentum * self.w_mom[i] -
                             self.rate * w_update)
            self.b_mom[i] = (self.momentum * self.b_mom[i] -
                             self.rate * b_update)
            # Update weights
            self.weights[i] += self.w_mom[i]
            self.biases[i] += self.b_mom[i]

    def _Xy_loss(self, X, y):
        # Get probabilities
        self._forward(X)
        y_ = self.act[-1]
        # Calculate loss
        return self._loss(y, y_)

    def _loss(self, y, y_):
        # Primary loss
        primary = np.sum(-y * np.log(y_))
        # L2 regularisation
        sq_weights_sum = 0
        # Every element of self.weights is a numpy array of weights for a layer
        for weight in self.weights:
            sq_weights_sum += np.sum(weight ** 2)
        reg = self.reg * sq_weights_sum / 2
        return (primary + reg) / y.shape[0]

    def _predict(self, X):
        # Get probabilites
        self._forward(X)
        y_ = self.act[-1]
        # Argmax over probabilities
        return np.argmax(y_, axis=1)

    def _Xy_accuracy(self, X, y):
        # Ground truth label
        true = np.argmax(y, axis=1)
        # Get predictions
        preds = self._predict(X)
        # Calculate accuracy
        return self._accuracy(true, preds)

    def _accuracy(self, true, preds):
        # Accuracy is average number of correct predictions
        return np.sum(preds == true) / true.size

    def _softmax(self, z):
        # Stabilise Softmax by adding constant in exponent
        # numpy doesn't like arrays of shape (n, )
        shiftz = z - np.max(z, axis=1).reshape(-1, 1)
        return np.exp(shiftz) / np.sum(np.exp(shiftz), axis=1).reshape(-1, 1)

    def _relu(self, z):
        # Activation function
        relu = np.zeros_like(z)
        relu[z > 0] = z[z > 0]
        return relu

    def _theta(self, z):
        # Derivative of ReLU
        theta = np.zeros_like(z)
        theta[z > 0] = 1
        return theta


if __name__ == "__main__":
    # Initialise
    nn = VanillaNN(num_in=784, num_hidden=[100, 50], num_out=10)

    # Load data
    nn.load_data(data_path=None, val_frac=.3, testing=False)

    # Do training
    nn.train(num_epochs=5, batch_size=32, learning_rate=0.0001,
             learning_rate_decay=0.99, momentum=0.98, regularisation=1e-5,
             do_momentum=True, learning_curve=True)

    # nn._grad_check()

    # Empty unless learning_curve=True
    train_losses = nn.train_losses
    val_losses = nn.val_losses

    # Calculate accuracies
    train_acc, val_acc, test_acc = nn.accuracy()
    print('\nAccuracies; train {:.3f}, val {:.3f}, test {:.3f}'.format(
        train_acc, val_acc, test_acc))

    # Plot learning curves
    plt.plot(train_losses, label='train loss')
    plt.plot(val_losses, label='val loss')
    plt.title('Learning curve')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()
