from typing import Tuple

import numba as nb
import numpy as np


@nb.njit(cache=True)
def sigmoid(x):
    for i in range(x.shape[0]):
        x[i] = 1 / (1 + np.exp(-5 * x[i]))
    return x


@nb.njit(cache=True)
def split_weights_array(array, moving_window_size, concept_count):
    w_matrix_offset = moving_window_size * concept_count
    v_matrix_offset = w_matrix_offset + concept_count ** 2
    u_matrix = array[:w_matrix_offset].reshape(-1, concept_count)
    w_matrix = array[w_matrix_offset:v_matrix_offset].reshape(-1, concept_count)
    v_matrix = array[v_matrix_offset:].reshape(2, -1)
    return u_matrix, w_matrix, v_matrix


@nb.njit(cache=True)
def predict_series_class_idx(
        membership_matrix, u_matrix, w_matrix, v_matrix, moving_window_size, moving_window_stride
):
    total_votes = np.zeros(2, dtype=np.int32)
    total_weights = np.zeros(2)

    concept_count = membership_matrix.shape[1]

    # It is important that these arrays are allocated only once.
    a_array = np.empty(concept_count)
    predicted_concept_membership_array = np.empty(concept_count)
    predicted_class_memberships = np.empty(2)

    for i in range(moving_window_size, membership_matrix.shape[0], moving_window_stride):

        for j in range(concept_count):
            a_array[j] = 0
            for k in range(moving_window_size):
                a_array[j] += (membership_matrix[i - k, j] * u_matrix[k, j])
        a_array = sigmoid(a_array)

        for j in range(concept_count):
            predicted_concept_membership_array[j] = 0
            for k in range(concept_count):
                predicted_concept_membership_array[j] += w_matrix[j, k] * a_array[k]
        predicted_concept_membership_array = sigmoid(
            predicted_concept_membership_array
        )

        for j in range(2):
            predicted_class_memberships[j] = 0
            for k in range(concept_count):
                predicted_class_memberships[j] += (
                        v_matrix[j, k] * predicted_concept_membership_array[k]
                )
        predicted_class_memberships = sigmoid(predicted_class_memberships)

        if predicted_class_memberships[0] >= predicted_class_memberships[1]:
            total_votes[0] += 1
        else:
            total_votes[1] += 1
        for j in range(2):
            total_weights[j] += predicted_class_memberships[j]

    if total_votes[0] < total_votes[1]:
        selected_idx = 1
    elif total_votes[0] > total_votes[1]:
        selected_idx = 0
    elif total_weights[0] < total_weights[1]:
        selected_idx = 1
    else:
        selected_idx = 0
    return selected_idx, total_weights


@nb.njit(cache=True)
def fitness_func(solution, membership_matrices, moving_window_size, moving_window_stride):
    u_matrix, w_matrix, v_matrix = split_weights_array(
        solution, moving_window_size, membership_matrices[0][0].shape[1]
    )
    misclassified_count = 0
    for idx in range(2):
        for membership_matrix in membership_matrices[idx]:
            assigned_class_idx = predict_series_class_idx(
                membership_matrix,
                u_matrix,
                w_matrix,
                v_matrix,
                moving_window_size,
                moving_window_stride,
            )[0]
            if assigned_class_idx != idx:
                misclassified_count += 1
    return misclassified_count
