from pyexpat import model
import preprocessing.cmeans_clustering as cmeans_clustering
from classification.binary_classifier_model import BinaryClassifierModel
import random
import math
import numpy as np
def create_asymetric_binary_classifiers(class_models,previous_considered_indices,move,concept_count):
    print("asymetric binary classifiers")
    binary_classifier_models = []
    for model1_idx, model1 in enumerate(class_models):
        for model2_idx, model2 in enumerate(class_models):
            if model1 != model2:
                centroids = model1[2][0]
                model1_memberships = model1[2][1]
                model2_memberships = cmeans_clustering.find_memberships(
                    model2[1], centroids
                )

                binary_classifier_models.append(
                    BinaryClassifierModel(
                        (model1_idx, model2_idx),
                        (model1_memberships, model2_memberships),
                        centroids,
                        previous_considered_indices,
                        move,
                    )
                )
    return binary_classifier_models

    
def create_symetric_binary_classifiers(class_models,previous_considered_indices,move,concept_count):
    print("symetric binary classifiers")
    binary_classifier_models = []

    for model1_idx, model1 in enumerate(class_models):
        for model2_idx, model2 in enumerate(class_models):
            if model1 != model2 and model1_idx<model2_idx:
                centroids = np.concatenate((model1[2][0], model2[2][0]))

                model1_memberships = cmeans_clustering.find_memberships(
                    model1[1], centroids
                )
                model2_memberships = cmeans_clustering.find_memberships(
                    model2[1], centroids
                )

                binary_classifier_models.append(
                    BinaryClassifierModel(
                        (model1_idx, model2_idx),
                        (model1_memberships, model2_memberships),
                        centroids,
                        previous_considered_indices,
                        move,
                    )
                )
    return binary_classifier_models
    
def create_combined_symetric_binary_classifiers(class_models,previous_considered_indices,move,concept_count):
    print("combined symetric binary classifiers")
    binary_classifier_models = []

    for model1_idx, model1 in enumerate(class_models):
        for model2_idx, model2 in enumerate(class_models):
            if model1 != model2 and model1_idx<model2_idx:
                centroids = cmeans_clustering.find_clusters(model1[1]+model2[1], concept_count=concept_count)
                
                class1_len = len(model1[1]) 
                binary_classifier_models.append(
                    BinaryClassifierModel(
                        (model1_idx, model2_idx),
                        (centroids[1][:class1_len],centroids[1][class1_len:]),
                        centroids[0],
                        previous_considered_indices,
                        move,
                    )
                )
    return binary_classifier_models
    
def create_k_vs_all_binary_classifiers(class_models,previous_considered_indices,move,concept_count):
    print("k vs all binary classifiers")
    binary_classifier_models = []
    reduce_fraction = 1/(len(class_models)-1)

    for model1_idx, model1 in enumerate(class_models):
        reduced_series = []
        for model2_idx, model2 in enumerate(class_models):
            if model1 != model2:
                reduced_size = math.ceil(reduce_fraction*len(model2[1]))
                reduced_series.extend(random.sample(model2[1], k=reduced_size))
        centroids = model1[2][0]
        model1_memberships = model1[2][1]
        rest_memberships = cmeans_clustering.find_memberships(
            reduced_series, centroids
        )

        binary_classifier_models.append(
            BinaryClassifierModel(
                (model1_idx, -1),
                (model1_memberships, rest_memberships),
                centroids,
                previous_considered_indices,
                move,
            )
        )
    return binary_classifier_models

