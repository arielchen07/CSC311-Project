from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse import csr_matrix, lil_matrix

from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, is_matrix, theta, beta, alpha):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param is_matrix: whether the input data is a sparse matrix, if false, then input is a dictionary
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector (how discriminative each question is)
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    if is_matrix:
        data_array = data.toarray()
        theta_minus_beta = theta[:, np.newaxis] - beta[np.newaxis, :]
        # print(theta_minus_beta)
        alpha_theta_minus_beta = theta_minus_beta * alpha
        sig = sigmoid(alpha_theta_minus_beta)
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
    else:
        log_lklihood = 0
        for i in range(len(data["is_correct"])):
            theta_minus_beta = theta[data["user_id"][i]] - beta[data["question_id"][i]]
            c_ij = data["is_correct"][i]
            alpha_theta_minus_beta = theta_minus_beta * alpha[data["question_id"][i]]
            sig = sigmoid(alpha_theta_minus_beta)
            log_lklihood += c_ij * np.log(sig) + (1 - c_ij) * np.log(1 - sig)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha):
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
    :param alpha: Vector (how discriminative each question is)
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    data_array = data.toarray()

    # data_array = np.array([[1, 0], [0, 1], [1, 1]])

    # fix beta, r, update theta
    theta_minus_beta_1 = theta[:, np.newaxis] - beta[np.newaxis, :]
    # print(theta_minus_beta_1)
    alpha_theta_minus_beta_1 = theta_minus_beta_1 * alpha
    # print(r_theta_minus_beta_1)
    sig_matrix1 = sigmoid(alpha_theta_minus_beta_1)
    # print(sig_matrix1)
    # partial_theta = np.sum(data_array, axis=1) - np.sum(sig_matrix1, axis=1)
    partial_theta = np.nansum((data_array - sig_matrix1) * alpha, axis=1)
    # print(partial_theta)
    # print(lr * partial_theta)
    theta += lr * partial_theta

    # fix theta, r, update beta
    theta_minus_beta_2 = theta[:, np.newaxis] - beta[np.newaxis, :]
    # print(theta_minus_beta_2)
    alpha_theta_minus_beta_2 = theta_minus_beta_2 * alpha
    # print(r_theta_minus_beta_2)
    sig_matrix2 = sigmoid(alpha_theta_minus_beta_2)
    # print(sig_matrix2)
    # partial_beta = - np.sum(data_array, axis=0) + np.sum(sig_matrix2, axis=0)
    partial_beta = np.nansum((-1 * data_array + sig_matrix2) * alpha, axis=0)
    # print(partial_theta)
    # print(lr * partial_theta)
    beta += lr * partial_beta

    # fix theta, beta, update r
    theta_minus_beta_3 = theta[:, np.newaxis] - beta[np.newaxis, :]
    # print(theta_minus_beta_3)
    alpha_theta_minus_beta_3 = theta_minus_beta_3 * alpha
    # print(r_theta_minus_beta_2)
    sig_matrix3 = sigmoid(alpha_theta_minus_beta_3)
    # print(sig_matrix3)
    partial_alpha = np.nansum((data_array - sig_matrix3) * theta_minus_beta_3, axis=0)
    # print(partial_alpha)
    # print(lr * partial_r)
    alpha += lr * partial_alpha
    r = np.clip(alpha, 0, 2)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


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
    theta = np.full(N, 0.5)
    beta = np.zeros(M)
    alpha = np.ones(M)

    train_log_like = []
    val_log_like = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, True, theta=theta, beta=beta, alpha=alpha)
        val_neg_lld = neg_log_likelihood(val_data, False, theta=theta, beta=beta, alpha=alpha)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        train_log_like.append(neg_lld)
        val_log_like.append(val_neg_lld)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, r = update_theta_beta(data, lr, theta, beta, alpha)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_log_like, val_log_like


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
    # hyperparameter:
    num_iteration = 80
    lr = 0.002

    theta, beta, acc, train_log_like, val_log_like = irt(sparse_matrix, val_data, lr, num_iteration)
    print("val accuracy: ")
    print(evaluate(val_data, theta, beta))
    # print("test accuracy: ")
    # print(evaluate(test_data, theta, beta))

    plt.plot([x for x in range(num_iteration)], train_log_like, label="train loglike")
    plt.xlabel("number of iterations")
    plt.ylabel("log likelihood")
    plt.title("Log likelihood vs iteration")
    plt.legend()
    plt.show()

    plt.plot([x for x in range(num_iteration)], val_log_like, label="val loglike")
    plt.xlabel("number of iterations")
    plt.ylabel("log likelihood")
    plt.title("Log likelihood vs iteration")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    plt.scatter(theta, [sigmoid(t - beta[4]) for t in theta], label="q1")
    plt.scatter(theta, [sigmoid(t - beta[2]) for t in theta], label="q2")
    plt.scatter(theta, [sigmoid(t - beta[3]) for t in theta], label="q3")
    plt.xlabel("theta")
    plt.ylabel("probability p(c_ij = 1)")
    plt.title("Probability vs Theta")
    plt.legend()
    plt.show()
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
