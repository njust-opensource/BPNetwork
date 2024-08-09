from BPNetwork.network import preview_data
import pandas as pd


if __name__ == '__main__':
    train_set = pd.read_csv('../dataset/mnist_train.csv', header=None)
    preview_data(train_set.iloc[4])
