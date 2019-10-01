import numpy as np
from _methods.utilities import shuffle_dataset
import pickle
import os

def ravel_image(data):
    dataset = []
    for bag in data:
        image_data = []
        for row in bag:
            for instance in row:
                image_data.append(instance)
        dataset.append(image_data)
    return dataset


def generate_positive_image(h, w, ratio, random_state):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        for k in range(h):
            r = random_state.randint(0, 255)
            b = random_state.randint(0, 255)
            image[k][i] = [r, 0, b]
    size = random_state.randint(int(w * ratio), w)
    image[:h][:size] = [0, 255, 0]
    return image, 1


def generate_negative_image(h, w, ratio, random_state):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        for k in range(h):
            r = random_state.randint(0, 255)
            b = random_state.randint(0, 255)
            image[k][i] = [r, 0, b]
    size = random_state.randint(0, int(w * (1 - ratio)))
    image[:h][:size] = [0, 0, 0]
    return image, -1


def get_dataset(random_seed, size=100, height=5, width=5, ratio=0.5):
    random_state = np.random.RandomState(seed=random_seed)
    features = []
    bag_labels = []
    for i in range(size):
        data, b_label = generate_positive_image(height, width, ratio, random_state)
        features.append(data)
        bag_labels.append(b_label)
        data, b_label = generate_negative_image(height, width, ratio, random_state)
        features.append(data)
        bag_labels.append(b_label)
    filepath = os.getcwd() + '/data/Image/image_set_{}_{}_{}_{}'.format(random_seed, size, height, width)
    image_file = open(filepath, 'wb')
    pickle.dump((features,bag_labels), image_file)
    features, bag_labels = shuffle_dataset(ravel_image(features), bag_labels, random_seed)
    return features, bag_labels
