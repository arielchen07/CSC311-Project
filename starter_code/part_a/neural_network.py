from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import matplotlib.pyplot as plt

from scipy.sparse import csc_matrix

# if the code is not running for you, try uncomment the line below (especially if you are using a Mac computer)
# from starter_code.utils import load_train_sparse, load_valid_csv, load_public_test_csv

def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.

        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)
        self.k = k

        self.encoder = torch.nn.Sequential(
            self.g,
            torch.nn.Sigmoid()
        )

        self.decoder = torch.nn.Sequential(
            self.h,
            torch.nn.Sigmoid()
        )

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        # reference: https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return decoded


def sigmoid(x):
    return 1 / (1 + torch.exp(-1 * x))


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    # reference: scipy.sparse.csc_matrix documentation
    valid_sparse = csc_matrix((valid_data["is_correct"], (valid_data["user_id"], valid_data["question_id"])),
                              shape=train_data.shape, dtype=np.int32).toarray()
    valid_sparse = torch.FloatTensor(valid_sparse)

    # count the number of non-zero entries, will be used to normalizing the squared loss
    N_valid = np.count_nonzero(valid_sparse)
    N_train = np.count_nonzero(zero_train_data)

    valid_acc_list = []
    train_cost_list = []
    valid_cost_list = []

    for epoch in range(0, num_epoch):
        train_loss = 0.
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target_train = inputs.clone()
            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask_train = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target_train[0][nan_mask_train] = output[0][nan_mask_train]

            loss = torch.sum(
                (output - target_train) ** 2) + 0.5 * lamb * model.get_weight_norm()  # with L2 regularization
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        loss_valid = valid_loss(model, zero_train_data, valid_data)
        valid_cost_list.append(loss_valid)
        valid_acc = evaluate(model, zero_train_data, valid_data)
        train_cost_list.append(train_loss)
        valid_acc_list.append(valid_acc)

        print("Epoch: {epoch}, Training loss: {train_loss}, "
              "Validation Loss: {valid_loss}, Valid Accuracy: {acc}".format(epoch=epoch,
                                                                            train_loss=train_loss,
                                                                            valid_loss=loss_valid, acc=valid_acc))
    epochs = np.arange(0, num_epoch, 1)
    plt.plot(epochs, np.array(valid_cost_list) / N_valid, label="average validation loss")
    plt.plot(epochs, np.array(train_cost_list) / N_train, label="average training loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Average training and validation loss")
    plt.legend()
    plt.show()
    plot(epochs, valid_acc_list, "validation accuracy", "validation accuracy vs number of epochs")
    plot(epochs, train_cost_list, "training loss", "training loss vs number of epochs")
    return valid_acc_list[-1]  # returns final validation accuracy
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def plot(x, y, ylabel, title, xlabel="number of epochs"):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        # print(output)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def valid_loss(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()
    loss = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item()
        loss += (guess - valid_data["is_correct"][i]) ** 2
    return loss


def to_sparse_matrix(val_data, train_matrix):
    N, M = train_matrix.shape
    val_sparse = np.zeros((N, M))
    for i in range(len(val_data["is_correct"])):
        val_sparse[val_data["user_id"][i], val_data["question_id"][i]] = val_data["is_correct"][i]
    return val_sparse


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_question = train_matrix.shape[1]
    # Set model hyperparameters.
    k_list = [10, 50, 100, 200, 500]
    lamb_list = [0.001, 0.01, 0.1, 1]
    k = k_list[0]

    # Set optimization hyperparameters (below are the hyperparameters tuned to optimal values).
    lr = 0.025
    num_epoch = 45
    lamb = lamb_list[0]

    model = AutoEncoder(num_question, k)
    final_valid_acc = train(model, lr, lamb, train_matrix, zero_train_matrix,
                            valid_data, num_epoch)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    print("Final test accuracy: {test}, Final valid accuracy: {valid}".format(test=test_acc, valid=final_valid_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
