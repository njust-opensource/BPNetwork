from BPNetwork.network import BPNNetwork, convert_data, preview_data
import pandas as pd
import numpy as np
import random

if __name__ == '__main__':
    selection = random.randint(0, 9999)
    test_set = pd.read_csv("../dataset/mnist_test.csv")
    xs, ys = convert_data(test_set)
    x = xs[selection].reshape(-1, 1)
    y = np.argmax(ys[selection], axis=0)
    network = BPNNetwork(input_size=784, hidden_size=128, output_size=10)
    network.load('../model/mnist_model-2024-08-09-01-01-48.npz')
    y_pre = network.predict(x)
    print(f"预测结果：{y_pre}")
    print(f"真实结果：{y}")
    preview_data(test_set.iloc[selection])
