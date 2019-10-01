import pickle
import os
import numpy as np
from _methods.utilities import shuffle_dataset


def get_dataset(random_seed):
    """
    Fetches dataset from precalculated descriptors.
    :return: features, bag_labels
    """
    list_of_positive_testing_bags = []
    list_of_negative_testing_bags = []
    list_of_positive_training_bags = []
    list_of_negative_training_bags = []
    features = []  # list for all bags (list containing instances)
    bag_labels = []  # list for all bag labels

    # Appends all pkl files to list of their
    for file in os.listdir("/home/dub/python/multiple_instance_learning/data/synthetic/testing/normal/"):
        if file.endswith('.pkl'):
            list_of_negative_testing_bags.append(
                os.path.join("/home/dub/python/multiple_instance_learning/data/synthetic/testing/normal/", file))
    for file in os.listdir('/home/dub/python/multiple_instance_learning/data/synthetic/testing/tumor/'):
        if file.endswith('.pkl'):
            list_of_positive_testing_bags.append(
                os.path.join("/home/dub/python/multiple_instance_learning/data/synthetic/testing/tumor/", file))
    for file in os.listdir('/home/dub/python/multiple_instance_learning/data/synthetic/training/normal/'):
        if file.endswith('.pkl'):
            list_of_negative_training_bags.append(
                os.path.join("/home/dub/python/multiple_instance_learning/data/synthetic/training/normal/", file))
    for file in os.listdir('/home/dub/python/multiple_instance_learning/data/synthetic/training/tumor/'):
        if file.endswith('.pkl'):
            list_of_positive_training_bags.append(
                os.path.join("/home/dub/python/multiple_instance_learning/data/synthetic/training/tumor/", file))

    # appends all instances and bag labels into lists
    for instance in list_of_positive_training_bags:  # positive instances
        with open(instance, 'rb') as tfh:
            tsd = pickle.load(tfh)
            if np.array(len(tsd['instances'][0])) == 16:
                features.append(np.array(tsd['instances']))
                bag_labels.append(tsd['bag_label'])

    for instance in list_of_negative_training_bags:  # negative instances
        with open(instance, 'rb') as tfh:
            tsd = pickle.load(tfh)
            if np.array(len(tsd['instances'][0])) == 16:
                features.append(np.array(tsd['instances']))
                bag_labels.append(-tsd['bag_label'])

    for instance in list_of_positive_testing_bags:  # positive instances
        with open(instance, 'rb') as tfh:
            tsd = pickle.load(tfh)
            if np.array(len(tsd['instances'][0])) == 16:
                features.append(np.array(tsd['instances']))
                bag_labels.append(-tsd['bag_label'])

    for instance in list_of_negative_testing_bags:  # negative instances
        with open(instance, 'rb') as tfh:
            tsd = pickle.load(tfh)
            if np.array(len(tsd['instances'][0])) == 16:
                features.append(np.array(tsd['instances']))
                bag_labels.append(tsd['bag_label'])
    features, bag_label = shuffle_dataset(features, bag_label, random_seed)
    return features, bag_labels
