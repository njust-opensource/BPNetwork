import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def convert_data(data: pd.DataFrame) -> tuple:
    y = data.iloc[:, 0].values
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    x = data.iloc[:, 1:].values.astype('float32') / 255.0
    return x, y_one_hot


def ReLU(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def ReLU_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    exps = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exps / np.sum(exps, axis=0, keepdims=True)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))


class BPNNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights = {
            'W_h': np.random.randn(self.hidden_size, self.input_size) * 0.01,
            'W_o': np.random.randn(self.output_size, self.hidden_size) * 0.01
        }

        self.biases = {
            'b_h': np.random.randn(self.hidden_size, 1),
            'b_o': np.random.randn(self.output_size, 1)
        }

    def forward_propagation(self, x_in: np.ndarray) -> dict:
        z_h = np.dot(self.weights['W_h'], x_in) + self.biases['b_h']
        x_h = ReLU(z_h)

        z_o = np.dot(self.weights['W_o'], x_h) + self.biases['b_o']
        y_hat = softmax(z_o)

        return {'z_h': z_h, 'x_h': x_h, 'z_o': z_o, 'y_hat': y_hat}

    def backward_propagation(self, x: np.ndarray, y_true: np.ndarray, forward: dict) -> dict:
        m = x.shape[0]

        dZ_o = forward['y_hat'] - y_true
        dW_o = np.dot(dZ_o, forward['x_h'].T)
        db_o = dZ_o

        dA_h = np.dot(self.weights['W_o'].T, dZ_o)
        dZ_h = dA_h * ReLU_derivative(forward['z_h'])
        dW_h = np.dot(dZ_h, x.T)
        db_h = dZ_h

        return {'dW_h': dW_h, 'db_h': db_h, 'dW_o': dW_o, 'db_o': db_o}

    def update_parameters(self, learning_rate: float, gradients: dict):
        self.weights['W_h'] -= learning_rate * gradients['dW_h']
        self.weights['W_o'] -= learning_rate * gradients['dW_o']
        self.biases['b_h'] -= learning_rate * gradients['db_h']
        self.biases['b_o'] -= learning_rate * gradients['db_o']

    def train(self, train_xs: np.ndarray, train_ys: np.ndarray, test_xs: np.ndarray, test_ys: np.ndarray,
              learning_rate: float, num_epochs: int) -> tuple:
        loss_history = pd.Series()
        last_loss = -1
        accuracy_history = pd.Series()

        for epoch in range(num_epochs):
            loss = -1
            for _ in range(len(train_xs)):
                x = train_xs[_].reshape(-1, 1)
                y_true = train_ys[_].reshape(-1, 1)
                forward = self.forward_propagation(x)
                loss = cross_entropy_loss(y_true, forward['y_hat'])
                gradients = self.backward_propagation(x, y_true, forward)
                self.update_parameters(learning_rate, gradients)

            counter = 0
            for _ in range(len(test_xs)):
                y_pre = self.predict(test_xs[_].reshape(-1, 1))
                if y_pre == np.argmax(test_ys[_], axis=0):
                    counter += 1
            accuracy_history[epoch] = counter / len(test_xs)

            if accuracy_history[epoch] > 0.97:
                print(f'Epoch {epoch}, Loss: {last_loss}, Accuracy: {accuracy_history[epoch]}')
                return loss_history, accuracy_history
            else:
                loss_history[epoch] = loss
                last_loss = loss

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {last_loss}, Accuracy: {accuracy_history[epoch]}')

    def predict(self, x: np.ndarray) -> np.ndarray:
        forward = self.forward_propagation(x)
        return np.argmax(forward['y_hat'], axis=0)[0]

    def save(self, path: str):
        np.savez(path, w_h=self.weights['W_h'], w_o=self.weights['W_o'], b_h=self.biases['b_h'], b_o=self.biases['b_o'])

    def load(self, path: str):
        data = np.load(path)
        self.weights['W_h'] = data['w_h']
        self.weights['W_o'] = data['w_o']
        self.biases['b_h'] = data['b_h']
        self.biases['b_o'] = data['b_o']


if __name__ == '__main__':
    train_set = pd.read_csv('../dataset/mnist_train.csv', header=None)
    test_set = pd.read_csv('../dataset/mnist_test.csv', header=None)

    train_x, train_y = convert_data(train_set)
    test_x, test_y = convert_data(test_set)

    network = BPNNetwork(input_size=784, hidden_size=128, output_size=10)
    loss_history, accuracy_history = network.train(train_x, train_y, test_x, test_y, learning_rate=0.01, num_epochs=1000)
    fig, ax =plt.subplots(2, 1)
    loss_history.plot(ax=ax[0], grid=True, label='Loss', style='--')
    accuracy_history.plot(ax=ax[1], grid=True, label='Accuracy')
    network.save(f"../model/mnist_model-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.npz")
    plt.legend()
    plt.show()
