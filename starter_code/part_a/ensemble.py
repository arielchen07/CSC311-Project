# TODO: complete this file.

from utils import *

import numpy as np
import scipy as sp
from sklearn.utils import resample
from sklearn.impute import KNNImputer


def knn_impute_by_user_pred(matrix, k):
    """

    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.toarray())
    return mat


def ensemble_3(matrix1, matrix2, matrix3):
    num_user = matrix1.shape[0]
    num_question = matrix1.shape[1]
    avg_matrix = []
    for i in range(num_user):
        avg_user = []
        for j in range(num_question):
            avg_pred = (matrix1[i, j] + matrix2[i, j] + matrix3[i, j]) / 3
            avg_user.append(avg_pred)
        avg_matrix.append(avg_user)
    return np.array(avg_matrix)


def main():
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # print(sparse_matrix)
    # print(sparse_matrix[0, 0])
    # first we resample three new set of training data
    train_1 = resample(sparse_matrix)
    print("Sparse matrix for the first train dataset:")
    print(train_1)
    print("Shape of sparse matrix:")
    print(train_1.shape)

    train_2 = resample(sparse_matrix)
    print("Sparse matrix for the first train dataset:")
    print(train_2)
    print("Shape of sparse matrix:")
    print(train_2.shape)

    train_3 = resample(sparse_matrix)
    print("Sparse matrix for the first train dataset:")
    print(train_3)
    print("Shape of sparse matrix:")
    print(train_3.shape)

    # use user-based with k = 11 based on the performance in Q1
    print("The validation accuracy of knn_impute_by_user on three base models:")
    train_1_imputed = knn_impute_by_user_pred(train_1, 11)
    train_2_imputed = knn_impute_by_user_pred(train_2, 11)
    train_3_imputed = knn_impute_by_user_pred(train_3, 11)

    train_1_sparse = sp.sparse.csr_matrix(train_1_imputed)
    train_2_sparse = sp.sparse.csr_matrix(train_2_imputed)
    train_3_sparse = sp.sparse.csr_matrix(train_3_imputed)

    ensemble_train = ensemble_3(train_1_sparse, train_2_sparse,
                                train_3_sparse)

    ensemble_val_acc = sparse_matrix_evaluate(val_data, ensemble_train)
    print(f'The averaged validation accuracy of three base models is'
          f' {ensemble_val_acc}')

    # performance on test data
    ensemble_test_acc = sparse_matrix_evaluate(test_data, ensemble_train)
    print(f'The averaged test accuracy of three base models is'
          f' {ensemble_test_acc}')


if __name__ == "__main__":
    main()
