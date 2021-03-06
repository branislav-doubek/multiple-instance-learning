import numpy as np
import scipy.io
from _methods.utilities import shuffle_dataset, train_test_split, into_dictionary, multiply_features
import os
import pickle

def get_dataset(args):
    """
    Loads and batches musk1 dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    if args.multiply:
        filepath = os.getcwd() + '/data/{}_dataset.pkl'.format(args.dataset)
    else:
        filepath = os.getcwd() + '/data/{}_original_dataset.pkl'.format(args.dataset)
    if (os.path.exists(filepath)):
        print('Dataset loaded')
        with open(filepath, 'rb') as dataset_file:
            dataset =  pickle.load(dataset_file)
            return dataset
    else:
        dataset = Dataset(args)
        print('Dataset loaded')
        file = open(filepath, 'wb')
        pickle.dump(dataset, file)
        return dataset

class Dataset():
    def __init__(self, args):
        """
        Loads and batches elephant dataset into feature and bag label lists
        :return: list(features), list(bag_labels)
        """
        self.rs = args.rs
        self.features = []
        self.bag_labels = []
        dataset = scipy.io.loadmat(os.getcwd() + '/data/musk1norm_matlab.mat')  # loads elephant dataset
        instance_bag_ids = np.array(dataset['bag_ids'])[0]
        instance_features = np.array(dataset['features'].todense())  # 1 feature point contains 230 dim vector
        if args.multiply:
            instance_features = multiply_features(instance_features)
        instance_labels = np.array(dataset['labels'].todense())[0]  # 100 positive and 100 bag negative labels
        bag_features = into_dictionary(instance_bag_ids,
                                       instance_features)
        bag_labels = into_dictionary(instance_bag_ids,
                                     instance_labels)  # creates dictionary whereas key is bag and values are instance
        for i in range(1, 93):
            self.features.append(np.array(bag_features.pop(i)))
            self.bag_labels.append(max(bag_labels[i]))
        self.random_shuffle()

    def random_shuffle(self):
        self.features, self.bag_labels = shuffle_dataset(self.features, self.bag_labels, self.rs)
        x_train, x_test, y_train, y_test = train_test_split(self.features, self.bag_labels)
        self.training_data = x_train
        self.testing_data = x_test
        self.training_labels = y_train
        self.testing_labels = y_test

    def return_training_set(self):
        return self.training_data, self.training_labels

    def return_testing_set(self):
        return self.testing_data, self.testing_labels

    def return_dataset(self):
        return self.features, self.bag_labels
