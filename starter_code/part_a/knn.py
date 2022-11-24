from matplotlib import pyplot as plt
from sklearn.impute import KNNImputer
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("By User Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T).T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("By Item Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    # print("Shape of valid date:")
    # print(val_data)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    # Q1(a)
    acc_dict = {}
    for k in [1, 6, 11, 16, 21, 26]:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_dict[k] = acc

    plt.figure()
    xs = acc_dict.keys()
    ys = [acc_dict[x] for x in xs]
    plt.scatter(xs, ys, s=15)
    plt.xlabel('k')
    plt.ylabel('accuracy on the validation data')
    plt.title("Accuracy on the validation data imputed by user "
              "as a function of k")
    plt.show()

    # Q1(b)
    print('final test accuracy with k=11')
    chosen_k = 11
    acc_test = knn_impute_by_user(sparse_matrix, test_data, chosen_k)

    # Q1(c)
    acc_dict = {}
    for k in [1, 6, 11, 16, 21, 26]:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_dict[k] = acc

    plt.figure()
    xs = acc_dict.keys()
    ys = [acc_dict[x] for x in xs]
    plt.scatter(xs, ys, s=15)
    plt.xlabel('k')
    plt.ylabel('accuracy on the validation data')
    plt.title("Accuracy on the validation data imputed by item "
              "as a function of k")
    plt.show()

    print('final test accuracy with k=21')
    chosen_k = 21
    acc_test = knn_impute_by_item(sparse_matrix, test_data, chosen_k)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
