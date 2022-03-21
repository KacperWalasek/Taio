"""
BinaryClassifierModel model (2 classification classes).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K


# https://stackoverflow.com/questions/56821382/how-to-restrict-weights-in-a-range-in-keras
class Between(Constraint):
    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


def create_model(concept_count, frame_size):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LocallyConnected2D(
                1,
                (frame_size, 1),
                use_bias=False,
                input_shape=(frame_size, concept_count, 1),
                activation="sigmoid",
                data_format="channels_last",
                kernel_constraint=Between(-1, 1),
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                concept_count,
                activation="sigmoid",
                use_bias=False,
                kernel_constraint=Between(-1, 1),
            ),
            tf.keras.layers.Dense(2, use_bias=False, kernel_constraint=Between(-1, 1),),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


class BinaryClassifierModel:
    """
    Class representing single class vs class classifier model.

    Parameters
    ----------
    class_numbers : tuple
        Tuple containing two class numbers.
    membership_matrices : tuple
        Tuple containing two lists of series membership matrices.
        Both elements of tuple correspond to classes specified in class_numbers.
    centroids: numpy.ndarray
        Ndarray of shape (concept_count, centroids_space_dimension) containing
        coordinates of centroids.
    previous_considered_indices: list
        List containing indices of previous elements which will be
        FCM's input for predicting the next one.
    move: int
        Step of frame used in processing single series.

    """

    def __init__(
        self,
        class_numbers,
        membership_matrices,
        centroids,
        previous_considered_indices,
        move,
    ):

        self.class_numbers = class_numbers
        self._membership_matrices = membership_matrices
        self.centroids = centroids
        self._previous_considered_indices = np.array(
            previous_considered_indices, dtype=np.int32
        )
        self._move = move

        self._concept_count = membership_matrices[0][0].shape[1]
        self.is_trained = False
        self._model = create_model(self._concept_count, len(previous_considered_indices))
        self._probability_model = None

    def train(self):
        """
        Method for training classifier instance.

        Returns
        -------
        None.

        """
        if self.is_trained:
            return self

        max_previous_index = self._previous_considered_indices.max()
        class_inputs_list = []
        for class_idx in range(2):
            inputs_list = []
            for membership_matrix in self._membership_matrices[class_idx]:
                input_indices = (
                    np.arange(
                        max_previous_index, membership_matrix.shape[0], self._move
                    )[:, np.newaxis].repeat(
                        self._previous_considered_indices.size, axis=1
                    )
                    - self._previous_considered_indices
                )
                inputs_list.append(membership_matrix[input_indices])
            class_inputs_list.append(np.vstack(inputs_list))

        x_train = np.expand_dims(np.vstack(class_inputs_list), 3)
        y_train = np.repeat(
            [0, 1], [class_inputs_list[0].shape[0], class_inputs_list[1].shape[0]]
        )

        dataset_train = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(x_train.shape[0])
            .batch(64)
        )

        stop_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

        print(f"Training binary classifier for classes {self.class_numbers}")
        self._model.fit(dataset_train, verbose=0, epochs=100, callbacks=[stop_callback], validation_split=0.1)
        print(f"Training binary classifier for classes ended {self.class_numbers}")

        self._probability_model = tf.keras.Sequential(
            [self._model, tf.keras.layers.Softmax()]
        )
        del self._model

        self.is_trained = True

        # good = 0
        # for matrix in self._membership_matrices[0]:
        #     pred = self.predict(matrix)
        #     if pred[0] == self.class_numbers[0]:
        #         good += 1
        # print(good / len(self._membership_matrices[0]))
        # good = 0
        # for matrix in self._membership_matrices[1]:
        #     pred = self.predict(matrix)
        #     if pred[0] == self.class_numbers[1]:
        #         good += 1
        # print(good / len(self._membership_matrices[1]))

        return self

    def predict(self, membership_matrix):
        """
        Method which classifies series to one of the classes specified
        in self.class_numbers tuple.

        Parameters
        ----------
        membership_matrix : numpy.ndarray
            Membership matrix of input series.

        Raises
        ------
        RuntimeError
            Classifier has not been trained.

        Returns
        -------
        tuple
            The first element is calculated class number and the second
            one is the sum of model's output weights for this class.

        """
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained")
        # pylint: disable = no-value-for-parameter
        max_previous_index = self._previous_considered_indices.max()
        input_indices = (
            np.arange(
                max_previous_index, membership_matrix.shape[0], self._move
            )[:, np.newaxis].repeat(
                self._previous_considered_indices.size, axis=1
            )
            - self._previous_considered_indices
        )
        x_test = np.expand_dims(membership_matrix[input_indices], 3)
        y_test = self._probability_model(x_test).numpy()

        total_probabilities = y_test.sum(axis = 0)
        return self.class_numbers[total_probabilities.argmax()], total_probabilities
