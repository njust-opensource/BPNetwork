from BPNetwork.network import convert_data, BPNNetwork
import matplotlib.pyplot as plt
import pandas as pd
import datetime

if __name__ == '__main__':
    train_set = pd.read_csv('../dataset/mnist_train.csv', header=None)
    test_set = pd.read_csv('../dataset/mnist_test.csv', header=None)

    train_x, train_y = convert_data(train_set)
    test_x, test_y = convert_data(test_set)

    network = BPNNetwork(input_size=784, hidden_size=128, output_size=10)
    loss_history, accuracy_history = network.train(train_x, train_y, test_x, test_y, learning_rate=0.01,
                                                   num_epochs=1000, target_accuracy=0.97)
    fig, ax = plt.subplots(2, 1)
    loss_history.plot(ax=ax[0], grid=True, label='Loss', style='--')
    accuracy_history.plot(ax=ax[1], grid=True, label='Accuracy')
    network.save(f"../model/mnist_model-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.npz")
    plt.legend()
    plt.show()
