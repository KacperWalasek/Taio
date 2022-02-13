import numba as nb
import numpy as np

@nb.njit#(nb.float64[:](nb.float64[:]))
def _sigmoid(x):
    return 1 / (1 + np.exp(-5 * x))

# https://stackoverflow.com/questions/20027936/how-to-efficiently-concatenate-many-arange-calls-in-numpy
# @nb.njit(nb.int32[:](nb.int32[:]))
# def _simple_multirange(counts):
#     # Remove the following line if counts is always strictly positive.
#     # counts = counts[counts != 0]
#     # TODO przemysl wyzej

#     counts1 = counts[:-1]
#     reset_index = np.cumsum(counts1)

#     incr = np.ones(counts.sum()).astype(np.int32)
#     incr[0] = 0
#     incr[reset_index] = 1 - counts1

#     # Reuse the incr array for the final result.
#     return incr.cumsum().astype(np.int32)

# @nb.njit(nb.types.UniTuple(nb.int32[:], 2)(nb.int32, nb.int32[:], nb.int32))
# def _multirange(start, stops, step):
#     """Example call: _multirange(1, np.r_[5, 4, 10], 3)

#     Args:
#         start ([type]): [description]
#         stops ([type]): [description]
#         step ([type]): [description]

#     Returns:
#         [type]: [description]
#     """
#     counts = np.ceil((stops - start)/step).astype(np.int32)
#     return (step * _simple_multirange(counts) + start, counts)

# @nb.njit(nb.int32[:](nb.int32[:], nb.int32, nb.float64[:, :], nb.int32[:], nb.float64[:, :], nb.float64[:, :], nb.float64[:, :]))
# def _predict_all_series(previous_considered_indices, move, membership_matrix, series_lengths, u_matrix, w_matrix, v_matrix):

#     max_previous_index = previous_considered_indices.max()

#     tmp, series_window_counts = _multirange(max_previous_index, series_lengths, move)
#     tmp[series_window_counts[0]:] += series_lengths.cumsum()[:-1].repeat(series_window_counts[1:])

#     # Note that targets for which we calculate predictions are
#     # membership_matrix[max_previous_index:membership_matrix.shape[0]].
#     # For every predicion we create an array containing indices
#     # of its input elements.
#     input_indices = (
#         np.expand_dims(tmp, axis = 1).repeat(previous_considered_indices.size).reshape(-1, previous_considered_indices.size)
#         - previous_considered_indices
#     )

#     # Now we compute matrix containing a-values for each target.
#     # The number of columns is equal to concept_count (each target element
#     # corresponds to single row which contains a-value for each concept).
#     test = membership_matrix[input_indices]
#     a_matrix = np.zeros(membership_matrix.shape[1])
#     for i in range(a_matrix.size):
#         for j in range(previous_considered_indices.size):
#             a_matrix[i] += membership_matrix[input_]

#     a_matrix = _sigmoid(
#         (membership_matrix[input_indices] * u_matrix).sum(axis=1)
#     )

#     # We calculate predicted values for each target.
#     predicted_concept_membership_matrix = _sigmoid(
#         np.matmul(a_matrix, w_matrix.T)
#     )

#     # The last step - class prediction.
#     # v_matrix must be of shape (2, concept_count)
#     predicted_class_memberships = _sigmoid(
#         (predicted_concept_membership_matrix[:, np.newaxis] * v_matrix).sum(axis=2)
#     )
#     discrete_class_memberships = np.zeros_like(
#         predicted_class_memberships
#     ).astype(np.int32)
#     # Below range is necessary.
#     discrete_class_memberships[
#         range(predicted_class_memberships.shape[0]),
#         predicted_class_memberships.argmax(1),
#     ] = 1
#     # series_windows_limits contains indices of first window for each series
#     series_windows_limits = np.zeros(series_lengths.size).astype(np.int32)
#     np.cumsum(series_window_counts[:-1], out = series_windows_limits[1:])

#     total_votes = np.add.reduceat(discrete_class_memberships, series_windows_limits, axis = 0)
#     total_weights = np.add.reduceat(predicted_class_memberships, series_windows_limits, axis = 0) / series_window_counts[:, np.newaxis]

#     ambiguous_rows = total_votes[:, 0] == total_votes[:, 1]
#     chosen_class_idx = total_votes.argmax(axis = 1)
#     chosen_class_idx[ambiguous_rows] = total_weights[ambiguous_rows].argmax(axis = 1)

#     return chosen_class_idx


@nb.njit#(nb.types.UniTuple(nb.float64[:,:], 3)(nb.float64[:], nb.int32, nb.int32))
def _split_uwv_array(array, previous_considered_indices_size, concept_count):
    w_matrix_offset = previous_considered_indices_size * concept_count
    v_matrix_offset = w_matrix_offset + concept_count ** 2
    u_matrix = array[:w_matrix_offset].reshape(-1, concept_count)
    w_matrix = array[w_matrix_offset:v_matrix_offset].reshape(
        -1, concept_count
    )
    v_matrix = array[v_matrix_offset:].reshape(2, -1)
    return u_matrix, w_matrix, v_matrix

@nb.njit#(nb.int32(nb.int32[:], nb.int32, nb.float64[:, :], nb.float64[:,:], nb.float64[:,:], nb.float64[:, :]))
def _predict_series_class_idx(previous_considered_indices, move, membership_matrix, u_matrix, w_matrix, v_matrix):
    max_previous_index = previous_considered_indices.max()
    
    total_votes = np.zeros(2, dtype = np.int32)
    total_weights = np.zeros(2)

    for i in range(max_previous_index, membership_matrix.shape[0], move):
        a_matrix = _sigmoid(np.ascontiguousarray((membership_matrix[i - previous_considered_indices] * u_matrix).sum(axis = 0)))
        predicted_concept_membership_matrix = _sigmoid(
            np.dot(w_matrix, a_matrix)
        )
        predicted_class_memberships = _sigmoid(
            (predicted_concept_membership_matrix * v_matrix).sum(axis=1)
        )
        total_votes[predicted_class_memberships.argmax()] += 1
        total_weights += predicted_class_memberships

    if total_votes[0] < total_votes[1]:
        return 1
    elif total_votes[0] > total_votes[1]:
        return 0
    elif total_weights[0] < total_weights[1]:
        return 1
    else:
        return 0

@nb.njit
def fitness_func(solution, membership_matrices, previous_considered_indices, move):

    u_matrix, w_matrix, v_matrix = _split_uwv_array(solution, previous_considered_indices.size,
        membership_matrices[0][0].shape[1])
    misclassified_count = 0
    for idx in range(2):
        for membership_matrix in membership_matrices[idx]:
            assigned_class_idx = _predict_series_class_idx(
                previous_considered_indices, move, membership_matrix, u_matrix, w_matrix, v_matrix
            )
            if assigned_class_idx != idx:
                misclassified_count += 1
    return misclassified_count

# def _predict_all_series(previous_considered_indices, move, membership_matrix, series_lengths, u_matrix, w_matrix, v_matrix):

#     max_previous_index = previous_considered_indices.max()

#     tmp, series_window_counts = _multirange(max_previous_index, series_lengths, move)
#     tmp[series_window_counts[0]:] += series_lengths.cumsum()[:-1].repeat(series_window_counts[1:])

#     # Note that targets for which we calculate predictions are
#     # membership_matrix[max_previous_index:membership_matrix.shape[0]].
#     # For every predicion we create an array containing indices
#     # of its input elements.
#     input_indices = (
#         np.expand_dims(tmp, axis = 1).repeat(previous_considered_indices.size, axis=1)
#         - previous_considered_indices
#     )

#     test = membership_matrix[input_indices]

#     # Now we compute matrix containing a-values for each target.
#     # The number of columns is equal to concept_count (each target element
#     # corresponds to single row which contains a-value for each concept).
#     a_matrix = _sigmoid(
#         (membership_matrix[input_indices] * u_matrix).sum(axis=1)
#     )

#     # We calculate predicted values for each target.
#     predicted_concept_membership_matrix = _sigmoid(
#         np.matmul(a_matrix, w_matrix.T)
#     )

#     # The last step - class prediction.
#     # v_matrix must be of shape (2, concept_count)
#     predicted_class_memberships = _sigmoid(
#         (predicted_concept_membership_matrix[:, np.newaxis] * v_matrix).sum(axis=2)
#     )
#     discrete_class_memberships = np.zeros_like(
#         predicted_class_memberships
#     ).astype(np.int32)
#     # Below range is necessary.
#     discrete_class_memberships[
#         range(predicted_class_memberships.shape[0]),
#         predicted_class_memberships.argmax(1),
#     ] = 1
#     # series_windows_limits contains indices of first window for each series
#     series_windows_limits = np.zeros(series_lengths.size).astype(np.int32)
#     np.cumsum(series_window_counts[:-1], out = series_windows_limits[1:])

#     total_votes = np.add.reduceat(discrete_class_memberships, series_windows_limits, axis = 0)
#     total_weights = np.add.reduceat(predicted_class_memberships, series_windows_limits, axis = 0) / series_window_counts[:, np.newaxis]

#     ambiguous_rows = total_votes[:, 0] == total_votes[:, 1]
#     chosen_class_idx = total_votes.argmax(axis = 1)
#     chosen_class_idx[ambiguous_rows] = total_weights[ambiguous_rows].argmax(axis = 1)

#     return chosen_class_idx