import numpy as np
import pandas as pd
import os
import pickle
from _methods.utilities import train_test_split, shuffle_dataset


def create_positive_instances(random_state, size=1000, mean=1, std=0.3):
    x = random_state.normal(mean, std, size=size)
    y = random_state.normal(mean, std, size=size)
    label = np.ones(size, dtype=int)
    return [list(a) for a in zip(x, y, label)]

def create_negative_instances(random_state, size=2000, mean=1, std=0.3):
    f_list = []
    combinations = [[-mean, mean], [mean, -mean], [-mean, -mean]]
    for mean_x, mean_y in combinations:
            x = random_state.normal(mean_x, std, size=size)
            y = random_state.normal(mean_y, std, size=size)
            label = np.zeros(size, dtype=int)
            f_list.extend([a for a in zip(x, y, label)])
    return f_list


def get_dataset(args):
    """
    Loads and batches elephant dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    dataset = Dataset(args)
    return dataset

class Dataset():
    def __init__(self, args):
        """
        Loads and batches elephant dataset into feature and bag label lists
        :return: list(features), list(bag_labels)
        """
        bag_size = 10
        random_state = np.random.RandomState(seed=args.rs)
        list_positive = create_positive_instances(random_state)
        list_negative = create_negative_instances(random_state)
        ds = pd.DataFrame(list_positive, columns=['X','Y','Class'])
        self.rs = args.rs
        ds = ds.append(pd.DataFrame(list_negative,columns=['X','Y','Class']), ignore_index=True)
        ds = ds.sample(frac=1.0, random_state=args.rs) # shuffles dataset
        positive_bags = []
        positive_ilabels = []
        negative_bags = []
        negative_ilabels = []
        self.bag_labels = []
        self.features = []
        self.instance_labels = []
        for g, df in ds.groupby(np.arange(len(ds)) // bag_size):
            df_instance_labels = df.Class.to_numpy()
            df = df.drop(['Class'], axis=1)
            if max(df_instance_labels) == 1: #Uncomment for MIMN definitition of bag
            #if sum(df_instance_labels) >= 0.5*bag_size:
                positive_bags.append(df.to_numpy())
                positive_ilabels.append(df_instance_labels)
            else:
                negative_bags.append(df.to_numpy())
                negative_ilabels.append(df_instance_labels)
        balanced_dataset = min(len(positive_bags), len(negative_bags), 100)
        self.features.extend(positive_bags[:balanced_dataset])
        self.bag_labels.extend([1 for _ in positive_bags[:balanced_dataset]])
        self.instance_labels.extend(positive_ilabels[:balanced_dataset])
        self.features.extend(negative_bags[:balanced_dataset])
        self.bag_labels.extend([-1 for _ in negative_bags[:balanced_dataset]])
        self.instance_labels.extend(negative_ilabels[:balanced_dataset])
        self.random_shuffle()

    def random_shuffle(self):
        self.features, self.bag_labels, self.instance_labels = shuffle_dataset(self.features, self.bag_labels, self.rs, self.instance_labels)
        x_train, x_test, y_train, y_test, iy_train, iy_test = train_test_split(self.features, self.bag_labels, self.instance_labels)
        self.training_data = x_train
        self.testing_data = x_test
        self.training_labels = y_train
        self.testing_labels = y_test
        self.training_instance_labels = iy_train
        self.testing_instance_labels = iy_test

    def return_name(self):
        return 'noise'

    def return_training_set(self):
        return self.training_data, self.training_labels

    def return_testing_set(self):
        return self.testing_data, self.testing_labels

    def return_dataset(self):
        return self.features, self.bag_labels

    def return_all_testing_data(self):
        return self.testing_data, self.testing_labels, self.testing_instance_labels
