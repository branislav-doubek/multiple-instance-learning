import numpy as np
from _methods.utilities import shuffle_dataset, train_test_split, ravel_image
import pickle
import os
import cv2


def generate_positive_image(h, w, random_state):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    rectangle_size = 6
    for i in range(w):
        for k in range(h):
            r = random_state.randint(0, 255)
            b = random_state.randint(0, 255)
            image[k][i] = [r, 0, b]
    y_min = int((h-rectangle_size)/2)
    x_min = int((w-rectangle_size)/2)
    y_max = int((h+rectangle_size)/2)
    x_max = int((w+rectangle_size)/2)
    im_arr_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(im_arr_bgr, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=-1)
    image = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)
    image_labels = get_instance_labels(image)
    return image, image_labels ,1


def generate_negative_image(h, w, random_state, max_rectangle_size=5):
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image_labels = np.zeros((h,w), dtype=np.uint8)
    for i in range(w):
        for k in range(h):
            r = random_state.randint(0, 255)
            b = random_state.randint(0, 255)
            image[k][i] = [r, 0, b]
    if max_rectangle_size != 0:
        rectangle_size = random_state.randint(max_rectangle_size)
        y_min = int((h-rectangle_size)/2)
        x_min = int((w-rectangle_size)/2)
        y_max = int((h+rectangle_size)/2)
        x_max = int((w+rectangle_size)/2)
        im_arr_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(im_arr_bgr, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=-1)
        image = cv2.cvtColor(im_arr_bgr, cv2.COLOR_BGR2RGB)
        image_labels = get_instance_labels(image)
    image_labels = get_instance_labels(image)
    return image, image_labels,  -1

def get_instance_labels(image):
    h, w = image.shape[0], image.shape[1]
    instance_labels = np.zeros((h,w))
    for h_i in range(h):
        for w_i in range(w):
            if (image[h_i, w_i] == np.array([  0, 255,   0])).all():
                instance_labels[h_i, w_i] = 1
            else:
                instance_labels[h_i, w_i] = 0
    return instance_labels

def get_dataset(args):
    """
    Loads and batches elephant dataset into feature and bag label lists
    :return: list(features), list(bag_labels)
    """
    dataset = Dataset(args)
    return dataset

class Dataset():
    def __init__(self, args):
        size=100
        self.h = 10
        self.w = 10
        self.rs = args.rs
        random_state = np.random.RandomState(seed=self.rs)
        self.images = []
        self.bag_labels = []
        self.instance_labels = []
        for i in range(size):
            data, instance_label, b_label = generate_positive_image(self.h, self.w, random_state)
            self.images.append(data)
            self.bag_labels.append(b_label)
            self.instance_labels.append(instance_label)
            #Uncomment for MIMN definition of positive image
            data, instance_label, b_label = generate_negative_image(self.h, self.w, random_state, 0)
            #data, instance_label, b_label = generate_negative_image(self.h, self.w, random_state)
            self.images.append(data)
            self.bag_labels.append(b_label)
            self.instance_labels.append(instance_label)
        self.features = ravel_image(self.images)
        self.instance_labels = ravel_image(self.instance_labels)
        self.features, self.bag_labels, self.instance_labels = shuffle_dataset(self.features, self.bag_labels, self.rs, self.instance_labels)
        self.random_shuffle()

    def return_name(self):
        return 'image'


    def random_shuffle(self):
        self.features, self.bag_labels, self.instance_labels = shuffle_dataset(self.features, self.bag_labels, self.rs, self.instance_labels)
        x_train, x_test, y_train, y_test, iy_train, iy_test = train_test_split(self.features, self.bag_labels, self.instance_labels)
        self.training_data = x_train
        self.testing_data = x_test
        self.training_labels = y_train
        self.testing_labels = y_test
        self.training_instance_labels = iy_train
        self.testing_instance_labels = iy_test

    def return_training_set(self):
        return self.training_data, self.training_labels

    def return_testing_set(self):
        return self.testing_data, self.testing_labels

    def return_dataset(self):
        return self.features, self.bag_labels

    def return_all_testing_data(self):
        return self.testing_data, self.testing_labels, self.testing_instance_labels

    def return_dimensions(self):
        return self.h, self.w
