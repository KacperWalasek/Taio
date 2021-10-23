"""
Module for training.
"""

# import os
# import read_data


# def train(train_dir):
#     """
#     Train classifier.

#     Parameters:
#     ----------
#     train_dir : string
#         Path to Train directory which contains directories (named with class numbers)
#         with CSV time series files.
#     """

#     i = 0

#     for dir_name, _, files in os.walk(train_dir):

#         if i == 0:
#             i = i + 1
#             continue

#         class_number = int(dir_name[-1])

#         for file in files:

#             file_name = os.path.join(dir_name, file)

#             data = read_data.process_data(file_name)
