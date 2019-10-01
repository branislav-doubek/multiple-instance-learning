import numpy as np
import scipy.io
from _methods.utilities import shuffle_dataset, into_dictionary


def get_dataset(random_seed):
    """
    Loads and batches elephant dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    features = []
    bag_label = []
    dataset = scipy.io.loadmat('/home/dub/python/multiple_instance_learning/data/Elephant/elephant_100x100_matlab.mat')  # loads elephant dataset
    instance_bag_ids = np.array(dataset['bag_ids'])[0]
    instance_features = np.array(dataset['features'].todense())  # 1 feature point contains 230 dim vector
    instance_labels = np.array(dataset['labels'].todense())[0]  # 100 positive and 100 bag negative labels
    bag_features = into_dictionary(instance_bag_ids,
                                   instance_features)
    bag_labels = into_dictionary(instance_bag_ids,
                                 instance_labels)  # creates dictionary whereas key is bag and values are instance
    # goes through whole dataset
    for i in range(1, 201):
        features.append(np.array(bag_features.pop(i)))
        bag_label.append(max(bag_labels[i]))
    features, bag_label = shuffle_dataset(features, bag_label, random_seed)

    return features, bag_label
