from matplotlib import pyplot as plt
import scipy.sparse
from scipy.sparse import csr_matrix, lil_matrix

from utils import *

import numpy as np
import item_response


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha, c):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :param alpha: Vector (how discriminative each question is)
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0
    for i in range(len(data["is_correct"])):
        theta_minus_beta = theta[data["user_id"][i]] - beta[data["question_id"][i]]
        c_ij = data["is_correct"][i]
        alpha_theta_minus_beta = theta_minus_beta * alpha[data["question_id"][i]]
        sig = sigmoid(alpha_theta_minus_beta)
        sig = c + (1 - c) * sig
        log_lklihood += c_ij * np.log(sig) + (1 - c_ij) * np.log(1 - sig)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta, alpha, c):
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
    :param c: parameter for random guess (probability of getting a question right by random guess)
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # fix beta, r, update theta
    partial_theta = np.zeros(theta.shape[0])
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]
        sig = sigmoid(alpha[question_id] * (theta[user_id] - beta[question_id]))
        partial_theta[user_id] += ((1 - c) * sig * (1 - sig) * alpha[question_id]) * \
                                  (c_ij / (c + (1 - c) * sig) - (1 - c_ij) / (1 - c - (1 - c) * sig))
    theta += lr * partial_theta

    # fix theta, r, update beta
    partial_beta = np.zeros(beta.shape[0])
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]
        sig = sigmoid(alpha[question_id] * (theta[user_id] - beta[question_id]))
        partial_beta[question_id] += ((1 - c) * sig * (1 - sig) * alpha[question_id]) * \
                                  ((- c_ij) / (c + (1 - c) * sig) + (1 - c_ij) / (1 - c - (1 - c) * sig))
    beta += lr * partial_beta

    # fix theta, beta, update r
    partial_alpha = np.zeros(alpha.shape[0])
    for i in range(len(data["is_correct"])):
        user_id = data["user_id"][i]
        question_id = data["question_id"][i]
        c_ij = data["is_correct"][i]
        sig = sigmoid(alpha[question_id] * (theta[user_id] - beta[question_id]))
        # partial_alpha[question_id] += (c_ij - sig) * (theta[user_id] - beta[question_id])
        partial_alpha[question_id] += ((1 - c) * sig * (1 - sig) * (theta[user_id] - beta[question_id])) * \
                                      (c_ij / (c + (1 - c) * sig) - (1 - c_ij) / (1 - c - (1 - c) * sig))
    alpha += lr * partial_alpha
    alpha = np.clip(alpha, 0, 2)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta, alpha


def irt(data, val_data, lr, iterations, c):
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
    theta = np.full(542, 0.5)
    beta = np.zeros(1774)
    alpha = np.ones(1774)

    train_log_like = []
    val_log_like = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha, c=c)
        val_neg_lld = neg_log_likelihood(val_data, theta=theta, beta=beta, alpha=alpha, c=c)
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, c=c)
        train_log_like.append(neg_lld)
        val_log_like.append(val_neg_lld)
        val_acc_lst.append(score)
        print("Iteration: {} \t NLLK: {} \t Score: {}".format(i, neg_lld, score))
        theta, beta, r = update_theta_beta(data, lr, theta, beta, alpha, c)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, val_acc_lst, train_log_like, val_log_like


def evaluate(data, theta, beta, alpha, c):
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
        # p_a = sigmoid(alpha[q] * x)
        p_a = c + (1 - c) * sigmoid(alpha[q] * x)
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
    c_1 = 0.25
    c_0 = 0

    # uncomment code below to see validation results for each hyperparameter combination
    # warning: this could take very long time
    # num_iterations = [100, 150, 200]
    # lr = [0.0025, 0.001, 0.005]
    # for itr in num_iterations:
    #     for rate in lr:
    #         print(f"num iteration = {itr}, learning rate = {rate}")
    #         theta, beta, alpha, acc, train_log_like, val_log_like = irt(train_data, val_data, rate, itr, c_1)
    #         print(f"val accuracy modified c = {c_1}: {evaluate(val_data, theta, beta, alpha, c_1)}")
    #         theta, beta, alpha, acc, train_log_like, val_log_like = irt(train_data, val_data, rate, itr, c_0)
    #         print(f"val accuracy modified c = {c_0}: {evaluate(val_data, theta, beta, alpha, c_0)}")
    #         theta, beta, acc, train_log_like, val_log_like = item_response.irt(train_data, val_data, rate, itr)
    #         print(f"val accuracy old: {item_response.evaluate(val_data, theta, beta)}")

    # hyperparameter:
    # hyperparameters for modified irt (c = 0.25): num iteration = 100, learning rate = 0.0025
    # hyperparameters for modified irt (c = 0): num iteration = 100, learning rate = 0.001
    # hyperparameters for original irt: num iteration = 150, learning rate = 0.001
    print(f"hyperparameters used: num iteration = 100, learning rate = 0.0025")
    theta, beta, alpha, acc, train_log_like_1, val_log_like_1 = irt(train_data, val_data, 0.0025, 100, c_1)
    print(f"val accuracy modified c = {c_1}: {evaluate(val_data, theta, beta, alpha, c_1)}")
    print(f"test accuracy modified c = {c_1}: {evaluate(test_data, theta, beta, alpha, c_1)}")
    print(f"hyperparameters used: num iteration = 100, learning rate = 0.001")
    theta, beta, alpha, acc, train_log_like_2, val_log_like_2 = irt(train_data, val_data, 0.001, 100, c_0)
    print(f"val accuracy modified c = {c_0}: {evaluate(val_data, theta, beta, alpha, c_0)}")
    print(f"test accuracy modified c = {c_0}: {evaluate(test_data, theta, beta, alpha, c_0)}")
    print(f"hyperparameters used: num iteration = 150, learning rate = 0.001")
    theta, beta, acc, train_log_like_3, val_log_like_3 = item_response.irt(train_data, val_data, 0.001, 150)
    print(f"val accuracy original: {item_response.evaluate(val_data, theta, beta)}")
    print(f"test accuracy original: {item_response.evaluate(test_data, theta, beta)}")

    plt.plot([x for x in range(100)], train_log_like_1, label="train neg-loglike, modified c = 0.25")
    plt.plot([x for x in range(100)], train_log_like_2, label="train neg-loglike, modified c = 0")
    plt.plot([x for x in range(150)], train_log_like_3, label="train neg-loglike, original")
    plt.xlabel("number of iterations")
    plt.ylabel("negative log likelihood")
    plt.title("Negative Log likelihood for Training Set")
    plt.legend()
    plt.show()

    plt.plot([x for x in range(100)], val_log_like_1, label="val neg-loglike, modified c = 0.25")
    plt.plot([x for x in range(100)], val_log_like_2, label="val neg-loglike, modified c = 0")
    plt.plot([x for x in range(150)], val_log_like_3, label="val neg-loglike, original")
    plt.xlabel("number of iterations")
    plt.ylabel("negative log likelihood")
    plt.title("Negative Log likelihood for Validation Set")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)
    plt.scatter(theta, [sigmoid(t - beta[2]) for t in theta], label="q2")
    plt.scatter(theta, [sigmoid(t - beta[3]) for t in theta], label="q3")
    plt.scatter(theta, [sigmoid(t - beta[4]) for t in theta], label="q4")
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
