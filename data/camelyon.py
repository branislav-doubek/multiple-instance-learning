import joblib
import numpy as np
import os
from _methods.utilities import shuffle_dataset, multiply_features
import pickle
import random

def get_dataset(args):
    """
    Loads and batches Camelyon dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    filepath = os.getcwd() + '/data/{}_dataset_{}'.format(args.dataset, args.rs)
    if (os.path.exists(filepath)):
        print('Dataset loaded')
        dataset =  joblib.load(filepath)
        return dataset
    else:
        dataset = Dataset(args)
        print('Dataset loaded')
        joblib.dump(dataset, filepath)
        return dataset

class Dataset():
    def __init__(self, args):
        """
        Loads and batches elephant dataset into feature and bag label lists
        :return: list(features), list(bag_labels)
        """
        list_of_positive_testing_bags = []
        list_of_negative_testing_bags = []
        list_of_positive_training_bags = []
        list_of_negative_training_bags = []
        self.training_data = []
        self.testing_data = []
        self.training_labels = []
        self.testing_labels = []
        self.rs = args.rs
        self.pos_bags = []
        self.neg_bags = []
        # Appends all pkl files to lists
        filepath = os.getcwd()
        for file in os.listdir(filepath + '/data/camelyon_features/testing/normal/'):
            if file.endswith('.pkl'):
                list_of_negative_testing_bags.append(os.path.join(filepath + "/data/camelyon_features/testing/normal/", file))
        for file in os.listdir(filepath + '/data/camelyon_features/testing/tumor/'):
            if file.endswith('.pkl'):
                list_of_positive_testing_bags.append(os.path.join(filepath + "/data/camelyon_features/testing/tumor/", file))
        for file in os.listdir(filepath + '/data/camelyon_features/training/normal/'):
            if file.endswith('.pkl'):
                list_of_negative_training_bags.append(os.path.join(filepath + "/data/camelyon_features/training/normal/", file))
        for file in os.listdir(filepath + '/data/camelyon_features/training/tumor/'):
            if file.endswith('.pkl'):
                list_of_positive_training_bags.append(os.path.join(filepath + "/data/camelyon_features/training/tumor/", file))


        # appends all instances and bag labels into lists
        for instance in list_of_positive_training_bags:  # positive instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.pos_bags.append(np.array(tsd['instances']))

        for instance in list_of_negative_training_bags:  # negative instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.neg_bags.append(np.array(tsd['instances']))

        for instance in list_of_positive_testing_bags:  # positive instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.pos_bags.append(np.array(tsd['instances']))

        for instance in list_of_negative_testing_bags:  # negative instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.neg_bags.append(np.array(tsd['instances']))
        self.random_shuffle()

    def random_shuffle(self):
        random.seed(self.rs)
        random.shuffle(self.pos_bags)
        random.shuffle(self.neg_bags)
        self.training_data = self.pos_bags[:int(len(self.pos_bags)/2)] + self.neg_bags[:int(len(self.neg_bags)/2)]
        self.training_labels = [1 for _ in range(int(len(self.pos_bags)/2))] + [-1 for _ in range(int(len(self.neg_bags)/2))]
        self.testing_data = self.pos_bags[int(len(self.pos_bags)/2):] + self.neg_bags[int(len(self.neg_bags)/2):]
        self.testing_labels = [1 for _ in range(int(len(self.pos_bags)/2))] + [-1 for _ in range(int(len(self.neg_bags)/2))]
        self.testing_data, self.testing_labels = shuffle_dataset(self.testing_data, self.testing_labels, self.rs)
        self.training_data, self.training_labels = shuffle_dataset(self.training_data, self.training_labels, self.rs)


    def return_testing_set(self):
        return self.testing_data, self.testing_labels

    def return_training_set(self):
        return self.training_data, self.training_labels
