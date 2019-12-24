import joblib
import numpy as np
import os
from _methods.utilities import shuffle_dataset, multiply_features
import pickle

def get_dataset(args):
    """
    Loads and batches Camelyon dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    filepath = os.getcwd() + '/data/{}_dataset'.format(args.dataset)
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
                    self.training_data.append(np.array(tsd['instances']))
                    self.training_labels.append(tsd['bag_label'])

        for instance in list_of_negative_training_bags:  # negative instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.training_data.append(np.array(tsd['instances']))
                    self.training_labels.append(-tsd['bag_label'])
        '''
        for instance in list_of_positive_testing_bags:  # positive instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.testing_data.append(np.array(tsd['instances']))
                    self.testing_labels.append(tsd['bag_label'])

        for instance in list_of_negative_testing_bags:  # negative instances
            with open(instance, 'rb') as tfh:
                tsd = pickle.load(tfh)
                if np.array(len(tsd['instances'][0])) == 32:
                    self.testing_data.append(np.array(tsd['instances']))
                    self.testing_labels.append(-tsd['bag_label'])
        '''
        self.training_data, self.training_labels = shuffle_dataset(self.training_data, self.training_labels, self.rs)
        #self.testing_data, self.testing_labels = shuffle_dataset(self.testing_data, self.testing_labels, self.rs)


    def return_training_set(self):
        return self.training_data, self.training_labels

    def return_testing_set(self):
        return self.testing_data, self.testing_labels

    def return_training_set(self):
        return self.training_data, self.training_labels
