# TODO: complete this file.

# This is the ensemble for Item Response Theory.
# For more description, please see our report.

from utils import *
from item_response import *

import numpy as np

# this statement is used to get same set of results every time we run.
# np.random.seed(311)


def generate_new_dataset(data):
    """
    Randomly generate new dataset that sampled from data with replacement.
    The new dataset has the same number of samples.

    :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
    """
    randlst = np.random.randint(len(data['is_correct']),
                                size=len(data['is_correct']))
    new_dataset = {'user_id': [], 'question_id': [], 'is_correct': []}

    for i in randlst:
        new_dataset['user_id'].append(data['user_id'][i])
        new_dataset['question_id'].append(data['question_id'][i])
        new_dataset['is_correct'].append(data['is_correct'][i])

    return new_dataset


def irt(data, val_data, lr, iterations):
    """ Train IRT model and return the prediction.

        :param data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
        :param val_data: A dictionary {user_id: list, question_id: list,
        is_correct: list}
        :param lr: float
        :param iterations: int
        :return: (theta, beta, val_acc_lst)
        """
    # train IRT model
    theta = np.full(542, 0.5)
    beta = np.zeros(1774)
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        print(
            "Iteration: {} \t NLLK: {} \t Score: {}".format(i, neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # predict
    pred = []
    for i, q in enumerate(val_data["question_id"]):
        u = val_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        if p_a >= 0.5:
            pred.append(1)
        else:
            pred.append(0)
        # pred.append(p_a >= 0.5)
    return pred


# def predict(data, theta, beta):
#     pred = []
#     for i, q in enumerate(data["question_id"]):
#         u = data["user_id"][i]
#         x = (theta[u] - beta[q]).sum()
#         p_a = sigmoid(x)
#         pred.append(p_a >= 0.5)
#     return pred


def ensemble_3(train_data, val_data, lr, iteration):
    """
    Select and train 3 base models with bootstrapping the
    """

    train_1 = generate_new_dataset(train_data)
    pred1 = irt(train_1, val_data, lr, iteration)

    train_2 = generate_new_dataset(train_data)
    pred2 = irt(train_2, val_data, lr, iteration)

    train_3 = generate_new_dataset(train_data)
    pred3 = irt(train_3, val_data, lr, iteration)

    # average predictions for val_data
    avg_pred = []
    for i in range(len(val_data['is_correct'])):
        avg = (pred1[i] + pred2[i] + pred3[i]) / 3
        avg_pred.append(avg >= 0.5)

    # evaluate accuracy
    return np.sum((val_data["is_correct"] == np.array(avg_pred))) \
           / len(val_data["is_correct"])


def main():
    # load train, validation, test data
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # initialize hyperparameter
    num_iteration = 150
    lr = 0.001

    # print(val_data['is_correct'])

    val_acc = ensemble_3(train_data, val_data, lr, num_iteration)
    print(f'The validation accuracy of ensemble is'
          f' {val_acc}')

    # performance on test data
    test_acc = ensemble_3(train_data, test_data, lr, num_iteration)
    print(f'The averaged test accuracy of ensemble is'
          f' {test_acc}')


if __name__ == "__main__":
    main()
