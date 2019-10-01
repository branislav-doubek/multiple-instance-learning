import numpy as np
import scipy.io
from _methods.utilities import shuffle_dataset


def into_dictionary(index, features):
    """
    helper function for transforming dataset from list to dict
    :param index: index of the instance [list]
    :param features: features  [list]
    :return: dictionary [dict]
    """
    dictionary = {}
    for index, value in zip(index, features):
        if index in dictionary:
            dictionary[index].append(value)
        else:
            dictionary[index] = [value]
    return dictionary


def get_dataset(random_seed):
    """
    Loads and batches fox dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    features = []
    bag_label = []
    dataset = scipy.io.loadmat('/home/dub/python/multiple_instance_learning/data/Fox/fox_100x100_matlab.mat')  # loads fox dataset
    instance_bag_ids = np.array(dataset['bag_ids'])[0]
    instance_features = np.array(dataset['features'].todense())  # 1 feature point contains 230 dim vector
    instance_labels = np.array(dataset['labels'].todense())[0]  # 100 positive and 100 negative bag labels
    bag_features = into_dictionary(instance_bag_ids,
                                   instance_features)  # creates dictionary whereas key is bag and values are
    # instance features
    bag_labels = into_dictionary(instance_bag_ids,
                                 instance_labels)  # creates dictionary whereas key is bag and values are instance
    for i in range(1, 201):  # goes through whole dataset
        features.append(np.array(bag_features.pop(i)))
        bag_label.append(max(bag_labels[i]))
    features, bag_label = shuffle_dataset(features, bag_label, random_seed)
    return features, bag_label
