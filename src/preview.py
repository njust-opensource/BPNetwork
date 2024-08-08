import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_set = pd.read_csv('../dataset/mnist_train.csv', header=None)

    sample = train_set.iloc[4]

    label = sample[0]
    pixels = sample[1:].values

    pixels = pixels.reshape(28, 28)

    plt.imshow(pixels, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
