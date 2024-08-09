from BPNetwork.network import BPNNetwork, convert_data, preview_data
import pandas as pd
import numpy as np
import random
import os

if __name__ == '__main__':
    best_accuracy = 0
    model_file = ''
    for file in os.listdir('../model'):
        l = file.rfind('-') + 1
        r = file.rfind('.')
        if float(file[l:r]) > best_accuracy:
            best_accuracy = float(file[l:r])
            model_file = file

    if model_file != '':
        print(f'Model selected: {model_file}')
        selection = random.randint(0, 9999)
        test_set = pd.read_csv("../dataset/mnist_test.csv")
        xs, ys = convert_data(test_set)
        x = xs[selection].reshape(-1, 1)
        y = np.argmax(ys[selection], axis=0)
        network = BPNNetwork(input_size=784, hidden_size=128, output_size=10)
        network.load(f'../model/{model_file}')

        y_pre = network.predict(x)
        print(f"预测结果：{y_pre}")
        print(f"真实结果：{y}")
        preview_data(test_set.iloc[selection])
    else:
        print('No model file selected.')
