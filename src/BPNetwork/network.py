import math

from matplotlib import pyplot as plt
from tqdm import trange , tqdm
import pandas as pd
import numpy as np


def preview_data(sample: pd.Series) -> None:
    label = sample[0]
    pixels = sample[1:].values

    pixels = pixels.reshape(28, 28)

    plt.imshow(pixels, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()


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

        self.best_accuracy = 0
        self.best = {
            'weights': {
                'W_h': np.random.randn(self.hidden_size, self.input_size) * 0.01,
                'W_o': np.random.randn(self.output_size, self.hidden_size) * 0.01
            },
            'bias': {
                'b_h': np.random.randn(self.hidden_size, 1),
                'b_o': np.random.randn(self.output_size, 1)
            }
        }

    def forward_propagation(self, x_in: np.ndarray) -> dict:
        z_h = np.dot(self.weights['W_h'], x_in) + self.biases['b_h']
        x_h = ReLU(z_h)

        z_o = np.dot(self.weights['W_o'], x_h) + self.biases['b_o']
        y_hat = softmax(z_o)

        return {'z_h': z_h, 'x_h': x_h, 'z_o': z_o, 'y_hat': y_hat}

    def backward_propagation(self, x: np.ndarray, y_true: np.ndarray, forward: dict) -> dict:

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
              learning_rate: float, num_epochs: int, target_accuracy: float = 0.97) -> tuple:
        loss_history = pd.Series()
        last_loss = -1
        accuracy_history = pd.Series()

        for epoch in range(num_epochs):
            try:
                loss = -1

                with trange(len(train_xs), desc="Training", unit=' Iter', ncols=150, position=0) as pbar:
                    for _ in pbar:
                        x = train_xs[_].reshape(-1, 1)
                        y_true = train_ys[_].reshape(-1, 1)
                        forward = self.forward_propagation(x)
                        loss = cross_entropy_loss(y_true, forward['y_hat'])
                        gradients = self.backward_propagation(x, y_true, forward)
                        self.update_parameters(learning_rate, gradients)

                        pbar.set_description(f'Epoch: {epoch:4d}, Last loss :{loss:+.2e}, '
                                             f'Best Accuracy: {self.best_accuracy:2.2%}')

                counter = 0
                for __ in range(len(test_xs)):
                    y_pre = self.predict(test_xs[__].reshape(-1, 1))
                    if y_pre == np.argmax(test_ys[__], axis=0):
                        counter += 1
                accuracy_history[epoch] = counter / len(test_xs)

                if accuracy_history[epoch] > self.best_accuracy:
                    self.best_accuracy = accuracy_history[epoch]
                    self.best['weights'] = self.weights
                    self.best['bias'] = self.biases

                if accuracy_history[epoch] > target_accuracy:
                    tqdm.write(f'Epoch {epoch + 1}, Last loss: {last_loss}, Best Accuracy: {self.best_accuracy}')
                    return loss_history, accuracy_history
                else:
                    loss_history[epoch] = loss
                    last_loss = loss
            except KeyboardInterrupt:
                tqdm.write(f"Interrupt from Keyboard. Best Accuracy: {self.best_accuracy}")
                return loss_history, accuracy_history

    def predict(self, x: np.ndarray) -> int:
        forward = self.forward_propagation(x)
        return np.argmax(forward['y_hat'], axis=0)[0]

    def save(self, path: str):
        np.savez(path,
                 w_h=self.best['weights']['W_h'],
                 w_o=self.best['weights']['W_o'],
                 b_h=self.best['bias']['b_h'],
                 b_o=self.best['bias']['b_o'])

    def load(self, path: str):
        data = np.load(path)
        self.weights['W_h'] = data['w_h']
        self.weights['W_o'] = data['w_o']
        self.biases['b_h'] = data['b_h']
        self.biases['b_o'] = data['b_o']
