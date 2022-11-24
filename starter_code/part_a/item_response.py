from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # N, M = data.shape
    data_array = data.toarray()
    theta_minus_beta = theta[:, np.newaxis] - beta[np.newaxis, :]
    # print(theta_minus_beta)
    sig = sigmoid(theta_minus_beta)
    # print(sig)
    log_sig = np.log(sig)
    # print(log_sig)
    log_1_minus_sig = np.log(1 - sig)
    # print(log_1_minus_sig)
    # print(log_sig.shape)
    sum_first_part = data_array * log_sig
    # print(sum_first_part)
    sum_second_part = (1 - data_array) * log_1_minus_sig
    # print(sum_second_part)
    log_lklihood = np.nansum(sum_first_part + sum_second_part)
    # print(log_lklihood)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    data_array = data.toarray()

    # fix beta, update theta
    theta_minus_beta_1 = theta[:, np.newaxis] - beta[np.newaxis, :]
    # print(theta_minus_beta_1)
    sig_matrix1 = sigmoid(theta_minus_beta_1)
    # print(sig_matrix1)
    # partial_theta = np.sum(data_array, axis=1) - np.sum(sig_matrix1, axis=1)
    partial_theta = np.nansum(data_array - sig_matrix1, axis=1)
    # print(partial_theta)
    # print(lr * partial_theta)
    theta += lr * partial_theta

    # fix theta, update beta
    theta_minus_beta_2 = theta[:, np.newaxis] - beta[np.newaxis, :]
    # print(theta_minus_beta_2)
    sig_matrix2 = sigmoid(theta_minus_beta_2)
    # print(sig_matrix2)
    # partial_beta = - np.sum(data_array, axis=0) + np.sum(sig_matrix2, axis=0)
    partial_beta = np.nansum(- data_array + sig_matrix2, axis=0)
    # print(partial_theta)
    # print(lr * partial_theta)
    beta += lr * partial_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    N, M = data.shape
    theta = np.zeros(N)
    beta = np.zeros(M)

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    theta, beta, acc = irt(sparse_matrix, val_data, 0.005, 15)
    print("val accuracy: ")
    print(evaluate(val_data, theta, beta))
    # print("test accuracy: ")
    # print(evaluate(test_data, theta, beta))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
